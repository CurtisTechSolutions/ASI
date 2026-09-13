package server

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

// Version of the Go server / CLI.
const Version = "0.1.0"

// ModelLabel is what the frontend shows for the active model kind.
const ModelLabel = "Count / reward (Go)"

// apiError is a client-visible failure with an HTTP status.
type apiError struct {
	status  int
	message string
}

func (e *apiError) Error() string { return e.message }

func badRequest(format string, args ...any) error {
	return &apiError{400, fmt.Sprintf(format, args...)}
}

func utcNow() string { return time.Now().UTC().Format("2006-01-02T15:04:05-07:00") }

func splitTexts(content, unit string, pageLines int) []string {
	return radixnet.SplitTexts(content, unit, pageLines)
}

// Job is one asynchronous operation (train / 2nrl / feedback), reported like
// the Python server's job status document.
type Job struct {
	mu         sync.Mutex
	ID         string
	Type       string
	State      string
	Progress   map[string]any
	History    []map[string]any
	Error      *string
	StartedAt  string
	FinishedAt *string
	stop       atomic.Bool
}

// Record remembers one epoch record.
func (j *Job) Record(rec map[string]any) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.History = append(j.History, rec)
	j.Progress = rec
}

// Finish closes the job.
func (j *Job) Finish(state string, err error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.State = state
	if err != nil {
		msg := err.Error()
		j.Error = &msg
	}
	now := utcNow()
	j.FinishedAt = &now
}

// Running reports whether the job is still running.
func (j *Job) Running() bool {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.State == "running"
}

// ToDict is the JSON status document.
func (j *Job) ToDict() map[string]any {
	j.mu.Lock()
	defer j.mu.Unlock()
	history := make([]map[string]any, len(j.History))
	copy(history, j.History)
	var errAny any
	if j.Error != nil {
		errAny = *j.Error
	}
	var finished any
	if j.FinishedAt != nil {
		finished = *j.FinishedAt
	}
	var progress any
	if j.Progress != nil {
		progress = j.Progress
	}
	return map[string]any{
		"id": j.ID, "type": j.Type, "state": j.State, "progress": progress, "history": history, "error": errAny,
		"started_at": j.StartedAt, "finished_at": finished, "stop_requested": j.stop.Load(),
	}
}

// Options configure a Service.
type Options struct {
	ModelPath     string
	Seed          int64
	Workers       int
	Exact         bool
	UploadDir     string
	CheckpointDir string
	Keep          int
	Quiet         bool
	Log           func(string)
	// OllamaURL / OllamaModel are the defaults of the /api/tutor endpoints
	// (empty: $OLLAMA_HOST and $RADIXNET_TUTOR_MODEL, else the local defaults).
	OllamaURL   string
	OllamaModel string
	// ChatGPTURL / ChatGPTModel are those of the hosted teacher (empty:
	// $OPENAI_BASE_URL and $RADIXNET_OPENAI_MODEL, else OpenAI's own); the key
	// always comes from the server's $OPENAI_API_KEY.
	ChatGPTURL   string
	ChatGPTModel string
}

// Service holds the model, the current job and the directories.  Readers
// (predict, score, status ...) take the read lock; a job holds the write lock
// and hands it over to waiting readers after every epoch.
type Service struct {
	mu        sync.RWMutex
	model     *radixnet.Model
	modelPath string
	seed      int64
	workers   int
	exact     bool
	jobMu     sync.Mutex
	job       *Job
	jobIDs    int
	uploads   *Uploads
	ckpts     *Checkpoints
	logf      func(string)
	started   time.Time
	// negative is the negative network this server filters with (nil until first used)
	negative *radixnet.Model
	// the teacher defaults of the tutor endpoints and the records of every tutor run
	ollamaURL    string
	ollamaModel  string
	chatgptURL   string
	chatgptModel string
	tutorHistory []map[string]any
	// epochDelay slows every epoch (tests: makes a job observable while running)
	epochDelay time.Duration
}

// NewService loads the model at opts.ModelPath when it exists, else creates a fresh one.
func NewService(opts Options) (*Service, error) {
	workers := opts.Workers
	if workers < 0 {
		workers = 0
	}
	radixnet.Workers = workers
	var m *radixnet.Model
	path := opts.ModelPath
	if path != "" {
		if _, err := os.Stat(path); err == nil {
			loaded, err := radixnet.Load(path)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", path, err)
			}
			m = loaded
		}
	}
	if m == nil {
		fresh, err := radixnet.NewModel(opts.Seed, radixnet.DefaultGraphOptions())
		if err != nil {
			return nil, err
		}
		m = fresh
	}
	m.Workers = workers
	m.G.Workers = workers
	m.Exact = opts.Exact
	s := &Service{model: m, modelPath: path, seed: opts.Seed, workers: workers, exact: opts.Exact, started: time.Now(), logf: opts.Log}
	s.ollamaURL = radixnet.DefaultOllamaURL()
	if url, err := radixnet.NormaliseOllamaURL(opts.OllamaURL); err == nil {
		s.ollamaURL = url
	}
	s.ollamaModel = radixnet.DefaultTutorModel()
	if name := strings.TrimSpace(opts.OllamaModel); name != "" {
		s.ollamaModel = name
	}
	s.chatgptURL = radixnet.DefaultChatGPTURL()
	if url, err := radixnet.NormaliseChatGPTURL(opts.ChatGPTURL); err == nil {
		s.chatgptURL = url
	}
	s.chatgptModel = radixnet.DefaultChatGPTModel()
	if name := strings.TrimSpace(opts.ChatGPTModel); name != "" {
		s.chatgptModel = name
	}
	if opts.UploadDir != "" {
		s.uploads = NewUploads(opts.UploadDir)
	}
	if opts.CheckpointDir != "" {
		s.ckpts = NewCheckpoints(opts.CheckpointDir, opts.Keep)
	}
	if s.logf == nil {
		s.logf = func(string) {}
	}
	return s, nil
}

// Model returns the active model (for tests).
func (s *Service) Model() *radixnet.Model { return s.model }

// -- jobs -------------------------------------------------------------------------------------------

func (s *Service) currentJob() *Job {
	s.jobMu.Lock()
	defer s.jobMu.Unlock()
	return s.job
}

func (s *Service) ensureIdle() error {
	if job := s.currentJob(); job != nil && job.Running() {
		return &apiError{409, fmt.Sprintf("a %s job (%s) is running; wait for it to finish or POST /api/job/stop", job.Type, job.ID)}
	}
	return nil
}

// startJob runs work on a goroutine holding the model's write lock; the
// progress hook releases it between epochs so readers get a turn.
func (s *Service) startJob(kind string, work func(job *Job, progress func(map[string]any), stop func() bool) error) (map[string]any, error) {
	s.jobMu.Lock()
	defer s.jobMu.Unlock()
	if s.job != nil && s.job.Running() {
		return nil, &apiError{409, fmt.Sprintf("a %s job (%s) is running; wait for it to finish or POST /api/job/stop", s.job.Type, s.job.ID)}
	}
	s.jobIDs++
	job := &Job{ID: fmt.Sprintf("%s-%d", kind, s.jobIDs), Type: kind, State: "running", StartedAt: utcNow(), History: []map[string]any{}}
	s.job = job
	go func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		progress := func(rec map[string]any) {
			job.Record(rec)
			if s.epochDelay > 0 {
				time.Sleep(s.epochDelay)
			}
			// hand the lock to waiting readers for a moment
			s.mu.Unlock()
			runtime.Gosched()
			s.mu.Lock()
		}
		stop := func() bool { return job.stop.Load() }
		var err error
		func() {
			defer func() {
				if r := recover(); r != nil {
					err = fmt.Errorf("panic: %v", r)
				}
			}()
			err = work(job, progress, stop)
		}()
		state := "done"
		if err != nil {
			state = "error"
			s.logf(fmt.Sprintf("%s job %s failed: %v", kind, job.ID, err))
		} else if job.stop.Load() {
			state = "stopped"
		}
		job.Finish(state, err)
	}()
	return job.ToDict(), nil
}

// JobStatus is the current / last job document (nil before the first one).
func (s *Service) JobStatus() map[string]any {
	job := s.currentJob()
	if job == nil {
		return nil
	}
	return job.ToDict()
}

// StopJob requests the current job to stop.
func (s *Service) StopJob() (map[string]any, error) {
	job := s.currentJob()
	if job == nil {
		return nil, &apiError{404, "no job has been started"}
	}
	job.stop.Store(true)
	return job.ToDict(), nil
}

// -- training and feedback --------------------------------------------------------------------------

// TrainRequest is the body of POST /api/train (learning rates are accepted and ignored).
type TrainRequest struct {
	Texts        []string
	Epochs       int
	AutoCompress bool
}

// StartTrain starts a train job over texts held in memory.
func (s *Service) StartTrain(texts []string, epochs int, autoCompress bool) (map[string]any, error) {
	return s.StartTrainSource(radixnet.SliceSource(texts), epochs, autoCompress, 0, false, 0)
}

// StartTrainSource starts a train job over a streaming source (uploads of any
// size stream through in chunks of chunkSize texts; 0 = the default).
func (s *Service) StartTrainSource(src radixnet.TextSource, epochs int, autoCompress bool, chunkSize int, parallelParts bool, inflight int) (map[string]any, error) {
	if epochs < 0 {
		return nil, badRequest("epochs must be >= 0, got %d", epochs)
	}
	return s.startJob("train", func(job *Job, progress func(map[string]any), stop func() bool) error {
		opts := radixnet.TrainOptions{Epochs: epochs, AutoCompress: autoCompress, Progress: progress, Stop: stop, ChunkSize: chunkSize, ParallelParts: parallelParts, Inflight: inflight}
		_, err := s.model.TrainSource(src, opts)
		return err
	})
}

// StartTwoNRL starts a 2nrl job: penalise bad, then count + reward good.
func (s *Service) StartTwoNRL(bad, good []string, negEpochs, posEpochs int, strength float64) (map[string]any, error) {
	return s.StartTwoNRLRated(bad, nil, good, nil, negEpochs, posEpochs, strength)
}

// StartTwoNRLRated is StartTwoNRL with a rating per text: badWeights scale the
// penalties and goodWeights the rewards, so each text is learned in proportion
// to how bad or how good it was rated (nil: every text alike).
func (s *Service) StartTwoNRLRated(
	bad []string, badWeights []float64, good []string, goodWeights []float64,
	negEpochs, posEpochs int, strength float64,
) (map[string]any, error) {
	return s.startJob("2nrl", func(job *Job, progress func(map[string]any), stop func() bool) error {
		return s.twoNRL(bad, badWeights, good, goodWeights, negEpochs, posEpochs, strength, progress, stop)
	})
}

func (s *Service) phase(texts []string, weights []float64, epochs int, count bool, reward float64, phase string, progress func(map[string]any), stop func() bool) error {
	opts := radixnet.TrainOptions{Epochs: epochs, AutoCompress: true, Phase: phase, Progress: progress, Stop: stop}
	var err error
	if count {
		_, err = s.model.RewardWeighted(texts, weights, opts, reward)
	} else {
		_, err = s.model.PunishWeighted(texts, weights, opts, reward)
	}
	return err
}

func (s *Service) twoNRL(bad []string, badWeights []float64, good []string, goodWeights []float64, negEpochs, posEpochs int, strength float64, progress func(map[string]any), stop func() bool) error {
	if err := s.phase(bad, badWeights, negEpochs, false, -math.Abs(strength), "negative", progress, stop); err != nil {
		return err
	}
	if stop != nil && stop() {
		return nil
	}
	if err := s.phase(good, goodWeights, posEpochs, true, math.Abs(strength), "positive", progress, stop); err != nil {
		return err
	}
	s.model.Meta["twonrl_runs"] = float64(s.model.MetaInt("twonrl_runs") + 1)
	return nil
}

// FeedbackAction is what rated texts lead to.
func FeedbackAction(good, bad []string) string {
	switch {
	case len(good) > 0 && len(bad) > 0:
		return "2nrl"
	case len(good) > 0:
		return "reward"
	case len(bad) > 0:
		return "punish"
	}
	return ""
}

// StartFeedback starts a feedback job from rated texts.
func (s *Service) StartFeedback(good, bad []string, negEpochs, posEpochs int, strength float64) (map[string]any, string, error) {
	return s.StartFeedbackRated(good, nil, bad, nil, negEpochs, posEpochs, strength)
}

// StartFeedbackRated is StartFeedback with a mark per text: goodWeights /
// badWeights (0..1) turn the thumbs into ratings, and every text is learned in
// proportion to its mark instead of every thumb counting alike.
func (s *Service) StartFeedbackRated(
	good []string, goodWeights []float64, bad []string, badWeights []float64,
	negEpochs, posEpochs int, strength float64,
) (map[string]any, string, error) {
	action := FeedbackAction(good, bad)
	if action == "" {
		return nil, "", badRequest("give 'good' (thumbs up) and/or 'bad' (thumbs down) texts: lists, good_text / bad_text (one per line) or good_files / bad_files (upload names)")
	}
	job, err := s.startJob("feedback", func(job *Job, progress func(map[string]any), stop func() bool) error {
		switch action {
		case "2nrl":
			return s.twoNRL(bad, badWeights, good, goodWeights, negEpochs, posEpochs, strength, progress, stop)
		case "reward":
			return s.phase(good, goodWeights, posEpochs, true, math.Abs(strength), "positive", progress, stop)
		default:
			return s.phase(bad, badWeights, negEpochs, false, -math.Abs(strength), "negative", progress, stop)
		}
	})
	return job, action, err
}

// -- synchronous operations -------------------------------------------------------------------------

func (s *Service) read(fn func(m *radixnet.Model) (any, error)) (any, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return fn(s.model)
}

func (s *Service) mutate(fn func(m *radixnet.Model) (any, error)) (any, error) {
	if err := s.ensureIdle(); err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureIdle(); err != nil {
		return nil, err
	}
	return fn(s.model)
}

// Kinds describes the one kind this server runs.
func Kinds() []map[string]any {
	return []map[string]any{{
		"kind": "count", "label": ModelLabel,
		"description": "edge weight = the edge's share of its node's traversals, all time and inside a sliding window, plus rewards - penalties; no learning rate; beam prediction with the top-K and bottom-K continuations (Go implementation)",
	}}
}

// Backends mirrors the Python status's backend availability block.
func (s *Service) Backends() map[string]any {
	return map[string]any{"python": false, "torch": false, "cuda": false, "mps": false, "go": true, "default": "go"}
}

// Status is GET /api/status.
func (s *Service) Status() (map[string]any, error) {
	out, err := s.read(func(m *radixnet.Model) (any, error) { return m.Stats(), nil })
	if err != nil {
		return nil, err
	}
	stats := out.(map[string]any)
	var job any
	if j := s.JobStatus(); j != nil {
		job = j
	}
	var modelPath, uploadDir, ckptDir any
	if s.modelPath != "" {
		modelPath, _ = filepath.Abs(s.modelPath)
	}
	if s.uploads != nil {
		uploadDir = s.uploads.Dir
	}
	if s.ckpts != nil {
		ckptDir = s.ckpts.Dir
	}
	stats["kind"] = "count"
	stats["model_label"] = ModelLabel
	stats["kinds"] = Kinds()
	stats["job"] = job
	stats["backends"] = s.Backends()
	stats["model_path"] = modelPath
	stats["checkpoint_dir"] = ckptDir
	stats["upload_dir"] = uploadDir
	stats["ollama"] = map[string]any{"url": s.ollamaURL, "model": s.ollamaModel}
	stats["chatgpt"] = map[string]any{
		"url": s.chatgptURL, "model": s.chatgptModel, "configured": radixnet.ChatGPTConfigured(),
	}
	stats["engine"] = "go"
	stats["workers"] = s.workers // 0 = no cap: one goroutine per text
	stats["goroutines"] = runtime.NumGoroutine()
	stats["counting"] = map[bool]string{true: "exact", false: "racy"}[s.exact]
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	stats["heap_bytes"] = ms.HeapAlloc
	stats["heap_sys_bytes"] = ms.Sys
	if limit := debug.SetMemoryLimit(-1); limit > 0 && limit < math.MaxInt64 {
		stats["memory_limit_bytes"] = limit
	} else {
		stats["memory_limit_bytes"] = nil
	}
	return stats, nil
}

// DescribeModel is GET /api/model.
func (s *Service) DescribeModel() (map[string]any, error) {
	out, err := s.read(func(m *radixnet.Model) (any, error) { return m.G.WeightConfig(), nil })
	if err != nil {
		return nil, err
	}
	var modelPath any
	if s.modelPath != "" {
		modelPath, _ = filepath.Abs(s.modelPath)
	}
	return map[string]any{
		"kind": "count", "label": ModelLabel, "kinds": Kinds(), "model_path": modelPath,
		"paths": map[string]any{"count": modelPath}, "in_memory": []string{"count"}, "weights": out, "engine": "go",
	}, nil
}

// SelectKind is POST /api/model/select: only the count kind exists here.
func (s *Service) SelectKind(kind string) (map[string]any, error) {
	if strings.ToLower(strings.TrimSpace(kind)) != "count" {
		return nil, badRequest("the Go server runs the count / reward model only (kind %q is served by the Python server)", kind)
	}
	desc, err := s.DescribeModel()
	if err != nil {
		return nil, err
	}
	stats, err := s.read(func(m *radixnet.Model) (any, error) { return m.Stats(), nil })
	if err != nil {
		return nil, err
	}
	desc["origin"] = "active"
	desc["stats"] = stats
	return desc, nil
}

// ConfigureWeights is POST /api/model/weights.
func (s *Service) ConfigureWeights(opts map[string]float64) (map[string]any, error) {
	out, err := s.mutate(func(m *radixnet.Model) (any, error) {
		if err := m.ConfigureWeights(opts); err != nil {
			return nil, badRequest("%v", err)
		}
		return map[string]any{"weights": m.G.WeightConfig(), "stats": m.Stats()}, nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// Invert flips every reward.
func (s *Service) Invert() (map[string]any, error) {
	out, err := s.mutate(func(m *radixnet.Model) (any, error) { m.Invert(); return m.Stats(), nil })
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// Compress merges unary chains.
func (s *Service) Compress() (map[string]any, error) {
	out, err := s.mutate(func(m *radixnet.Model) (any, error) {
		merges := m.G.Compress()
		stats := m.Stats()
		stats["merges"] = merges
		return stats, nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// Save writes the model (default: the server's model path); allowed while a job runs.
func (s *Service) Save(path string) (map[string]any, error) {
	target := path
	if target == "" {
		target = s.modelPath
	}
	if target == "" {
		return nil, badRequest("no 'path' given and the server was started without a model path")
	}
	abs, err := filepath.Abs(target)
	if err != nil {
		return nil, err
	}
	_, err = s.read(func(m *radixnet.Model) (any, error) { return nil, m.Save(abs) })
	if err != nil {
		return nil, err
	}
	st, err := os.Stat(abs)
	if err != nil {
		return nil, err
	}
	out := map[string]any{"path": abs, "bytes": st.Size(), "negative": nil}
	// the negative network is a second file beside the model: saving the work means saving both
	if path == "" && s.negative != nil && s.negative.G.Neg != nil && s.negative.G.Neg.TotalBlame > 0 {
		if side := s.negativePath(); side != "" {
			if saved, err := s.NegativeSave(side); err == nil {
				out["negative"] = saved
			}
		}
	}
	return out, nil
}

func (s *Service) replaceModel(m *radixnet.Model) (map[string]any, error) {
	m.Workers = s.workers
	m.G.Workers = s.workers
	m.Exact = s.exact
	out, err := s.mutate(func(_ *radixnet.Model) (any, error) {
		s.model = m
		return m.Stats(), nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// Load replaces the model with the one at path.
func (s *Service) Load(path string) (map[string]any, error) {
	if err := s.ensureIdle(); err != nil {
		return nil, err
	}
	m, err := radixnet.Load(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, &apiError{404, fmt.Sprintf("%s: no such file", path)}
		}
		return nil, badRequest("%v", err)
	}
	return s.replaceModel(m)
}

// Reset replaces the model with a fresh one.
func (s *Service) Reset(seed *int64, kind string, opts map[string]float64) (map[string]any, error) {
	if kind != "" && strings.ToLower(kind) != "count" {
		return nil, badRequest("the Go server runs the count / reward model only (kind %q is served by the Python server)", kind)
	}
	if err := s.ensureIdle(); err != nil {
		return nil, err
	}
	g := radixnet.DefaultGraphOptions()
	for name, v := range opts {
		switch name {
		case "count_scale":
			g.CountScale = v
		case "global_scale":
			g.GlobalScale = v
		case "window_scale":
			g.WindowScale = v
		case "reward_scale":
			g.RewardScale = v
		case "window":
			g.Window = int(v)
		}
	}
	useSeed := s.seed
	if seed != nil {
		useSeed = *seed
	}
	m, err := radixnet.NewModel(useSeed, g)
	if err != nil {
		return nil, badRequest("%v", err)
	}
	return s.replaceModel(m)
}

// Checkpoints is GET /api/checkpoints.
func (s *Service) Checkpoints() map[string]any {
	if s.ckpts == nil {
		return map[string]any{"checkpoints": []any{}, "latest": nil}
	}
	var latest any
	if rec := s.ckpts.Latest(); rec != nil {
		latest = rec
	}
	return map[string]any{"checkpoints": s.ckpts.List(), "latest": latest}
}

func (s *Service) requireCheckpoints() (*Checkpoints, error) {
	if s.ckpts == nil {
		return nil, badRequest("checkpoints require a checkpoint directory; start the server with --checkpoint-dir")
	}
	return s.ckpts, nil
}

// SaveCheckpoint checkpoints the model (step = total epochs, metrics = last record).
func (s *Service) SaveCheckpoint(tag string) (map[string]any, error) {
	ck, err := s.requireCheckpoints()
	if err != nil {
		return nil, err
	}
	if tag == "" {
		tag = "manual"
	}
	out, err := s.read(func(m *radixnet.Model) (any, error) {
		var metrics map[string]any
		if len(m.History) > 0 {
			metrics = m.History[len(m.History)-1]
		}
		return ck.Save(m, int(m.MetaInt("epochs_total")), tag, metrics)
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// RestoreCheckpoint loads a checkpoint by name.
func (s *Service) RestoreCheckpoint(name string) (map[string]any, error) {
	ck, err := s.requireCheckpoints()
	if err != nil {
		return nil, err
	}
	if err := s.ensureIdle(); err != nil {
		return nil, err
	}
	path, err := ck.Resolve(name)
	if err != nil {
		return nil, err
	}
	m, err := radixnet.Load(path)
	if err != nil {
		return nil, badRequest("%v", err)
	}
	return s.replaceModel(m)
}

// History is GET /api/history.
func (s *Service) History() (map[string]any, error) {
	out, err := s.read(func(m *radixnet.Model) (any, error) {
		hist := make([]map[string]any, len(m.History))
		copy(hist, m.History)
		return map[string]any{"history": hist}, nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// Graph is GET /api/graph: the top-limit nodes by visit count plus START / END and the edges among them.
func (s *Service) Graph(limit int) (map[string]any, error) {
	if limit < 0 {
		return nil, badRequest("'limit' must be >= 0 (got %d)", limit)
	}
	out, err := s.read(func(m *radixnet.Model) (any, error) { return graphView(m.G, limit), nil })
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

func graphView(g *radixnet.Graph, limit int) map[string]any {
	g.Prepare()
	real := []int{}
	for i := 2; i < len(g.Labels); i++ {
		if g.Alive[i] {
			real = append(real, i)
		}
	}
	top := real
	if limit < len(real) {
		sorted := append([]int(nil), real...)
		sort.Slice(sorted, func(a, b int) bool {
			if g.Count[sorted[a]] != g.Count[sorted[b]] {
				return g.Count[sorted[a]] > g.Count[sorted[b]]
			}
			return sorted[a] < sorted[b]
		})
		top = sorted[:limit]
	}
	ids := append([]int{radixnet.Start, radixnet.End}, top...)
	sort.Ints(ids[2:])
	chosen := make(map[int]bool, len(ids))
	for _, id := range ids {
		chosen[id] = true
	}
	nodes := make([]map[string]any, 0, len(ids))
	for _, i := range ids {
		nodes = append(nodes, map[string]any{"id": i, "label": g.Labels[i], "count": g.Count[i], "activation": 1.0, "z": 0.0, "a": 0.0, "b": 1.0 / 3.0, "h": 0.0, "k": 1.0})
	}
	edges := []map[string]any{}
	for _, p := range ids {
		shares := map[int]radixnet.Share{}
		for _, sh := range g.Shares(p) {
			shares[sh.Edge] = sh
		}
		for _, cc := range g.ChildCosts(p) {
			if !chosen[cc.Child] {
				continue
			}
			sh := shares[cc.Edge]
			edges = append(edges, map[string]any{
				"source": p, "target": cc.Child, "weight": g.EdgeW[cc.Edge], "count": g.EdgeCount[cc.Edge],
				"prob": math.Exp(-cc.Cost), "cost": cc.Cost, "reward": g.EdgeReward[cc.Edge],
				"share": sh.All, "recent_share": sh.Recent, "recent_count": g.WindowEdgeCount[cc.Edge],
			})
		}
	}
	sort.Slice(edges, func(i, j int) bool {
		if edges[i]["source"].(int) != edges[j]["source"].(int) {
			return edges[i]["source"].(int) < edges[j]["source"].(int)
		}
		return edges[i]["target"].(int) < edges[j]["target"].(int)
	})
	return map[string]any{
		"nodes": nodes, "edges": edges, "limit": limit, "total_nodes": g.NumNodes(), "total_edges": g.NumEdges(),
		"total_traversals": g.TotalTraversals, "window_traversals": g.WindowTraversals(), "window": g.WindowSize,
	}
}

// Predict / Generate / Score / Converse run under the read lock.

func (s *Service) Predict(prefix string, o radixnet.PredictOptions) (*radixnet.Prediction, error) {
	out, err := s.read(func(m *radixnet.Model) (any, error) { return m.Predict(prefix, o) })
	if err != nil {
		return nil, badRequest("%v", err)
	}
	return out.(*radixnet.Prediction), nil
}

func (s *Service) Generate(o radixnet.GenerateOptions) ([]*radixnet.PathResult, error) {
	out, err := s.read(func(m *radixnet.Model) (any, error) { return m.Generate(o) })
	if err != nil {
		return nil, badRequest("%v", err)
	}
	return out.([]*radixnet.PathResult), nil
}

func (s *Service) Score(text string) radixnet.Score {
	out, _ := s.read(func(m *radixnet.Model) (any, error) { return m.Score(text), nil })
	return out.(radixnet.Score)
}

func (s *Service) Converse(opening string, o radixnet.ConverseOptions) ([]*radixnet.Turn, error) {
	out, err := s.read(func(m *radixnet.Model) (any, error) { return m.Converse(opening, o) })
	if err != nil {
		return nil, badRequest("%v", err)
	}
	return out.([]*radixnet.Turn), nil
}

// UploadTexts reads training texts from uploads (400 when uploads are disabled).
func (s *Service) UploadTexts(names []string, unit string, pageLines int, wholeFile bool) ([]string, error) {
	if s.uploads == nil {
		return nil, badRequest("uploads are disabled: start the server with --upload-dir")
	}
	return s.uploads.Texts(names, unit, pageLines, wholeFile)
}

// UploadSource is a streaming source over uploads (400 when uploads are disabled).
func (s *Service) UploadSource(names []string, unit string, pageLines int, wholeFile bool) (radixnet.TextSource, error) {
	if s.uploads == nil {
		return nil, badRequest("uploads are disabled: start the server with --upload-dir")
	}
	return s.uploads.Source(names, unit, pageLines, wholeFile)
}
