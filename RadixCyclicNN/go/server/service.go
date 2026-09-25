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

// WordModelLabel is the same for a server running the word model.
const WordModelLabel = "Word n-gram (Go)"

// apiError is a client-visible failure with an HTTP status.
type apiError struct {
	status  int
	message string
}

func (e *apiError) Error() string { return e.message }

func badRequest(format string, args ...any) error {
	return &apiError{400, fmt.Sprintf(format, args...)}
}

func notFound(format string, args ...any) error {
	return &apiError{404, fmt.Sprintf(format, args...)}
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
	ModelPath string
	Seed      int64
	Workers   int
	Exact     bool
	UploadDir string
	// Encoding is how a model created here reads text (a model loaded from
	// ModelPath brings its own); the zero value is the character trigram.
	Encoding      radixnet.Encoding
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
	// Tools are the defaults of the /api/tools and /api/agent endpoints: what
	// the network may call, and how far it may reach.
	Offline      bool    // no web tools at all
	AllowPrivate bool    // let the web tools reach private addresses
	SearchURL    string  // a different search endpoint
	WebTimeout   float64 // seconds per web request (0: 20)
	MaxBytes     int     // cap on a fetched page (0: 2 MB)
	PythonTool   bool    // also offer the sandboxed `python` tool
	SandboxTime  float64 // seconds a sandboxed program may run (0: 10)
	NoIsolation  bool    // do not run sandboxed programs in their own network namespace
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
	// negMu guards that lazy load: the output paths reach it under the read lock
	negMu sync.Mutex
	// guardConfig is how strictly the negative network guards the output paths
	guardConfig radixnet.FilterConfig
	// the teacher defaults of the tutor endpoints and the records of every tutor run
	ollamaURL      string
	ollamaModel    string
	chatgptURL     string
	chatgptModel   string
	tutorHistory   []map[string]any
	criticHistory  []map[string]any
	codegenHistory []map[string]any
	agentHistory   []map[string]any
	chatHistory    []map[string]any
	// tools are the server's defaults for the external tools the network may call
	tools         toolDefaults
	evolveHistory []map[string]any
	// discriminator is the critic of the evolve loop (nil until first used)
	discriminator *radixnet.Model
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
		graph := radixnet.DefaultGraphOptions()
		graph.Encoding = opts.Encoding.WithDefaults()
		if err := graph.Encoding.Validate(); err != nil {
			return nil, err
		}
		fresh, err := radixnet.NewModel(opts.Seed, graph)
		if err != nil {
			return nil, err
		}
		m = fresh
	}
	m.Workers = workers
	m.G.Workers = workers
	m.Exact = opts.Exact
	tools := defaultToolDefaults()
	tools.Offline, tools.AllowPrivate, tools.PythonTool = opts.Offline, opts.AllowPrivate, opts.PythonTool
	tools.SearchURL, tools.NetworkIsolat = opts.SearchURL, !opts.NoIsolation
	if opts.WebTimeout > 0 {
		tools.WebTimeout = opts.WebTimeout
	}
	if opts.MaxBytes > 0 {
		tools.MaxBytes = opts.MaxBytes
	}
	if opts.SandboxTime > 0 {
		tools.SandboxTime = opts.SandboxTime
	}
	s := &Service{model: m, modelPath: path, seed: opts.Seed, workers: workers, exact: opts.Exact,
		started: time.Now(), logf: opts.Log, guardConfig: radixnet.DefaultFilterConfig(), tools: tools}
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
	return s.StartTrainPlanned(src, epochs, autoCompress, chunkSize, parallelParts, inflight, radixnet.Plan{})
}

// StartTrainPlanned is StartTrainSource with a training plan: the order, the
// curriculum, the rehearsal of the replay buffer and the early stop
// (../../SPEC-SearchAndTraining.md).  A plan out of range is a 400 before any
// job starts.
func (s *Service) StartTrainPlanned(src radixnet.TextSource, epochs int, autoCompress bool, chunkSize int, parallelParts bool, inflight int, plan radixnet.Plan) (map[string]any, error) {
	if epochs < 0 {
		return nil, badRequest("epochs must be >= 0, got %d", epochs)
	}
	if err := plan.Check(); err != nil {
		return nil, badRequest("%v", err)
	}
	return s.startJob("train", func(job *Job, progress func(map[string]any), stop func() bool) error {
		opts := radixnet.TrainOptions{Epochs: epochs, AutoCompress: autoCompress, Progress: progress, Stop: stop, ChunkSize: chunkSize, ParallelParts: parallelParts, Inflight: inflight, Plan: plan}
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
	s.model.MetaAddInt("twonrl_runs", 1)
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
//
// Words are not a kind: they are an encoding (`--encoding word:3:1`), so the
// units below follow the running model's encoding rather than its kind.
func Kinds(units string) []map[string]any {
	return []map[string]any{{
		"kind": "count", "label": ModelLabel, "units": units,
		"description": "edge weight = the edge's share of its node's traversals, all time and inside a sliding window, plus rewards - penalties; no learning rate; beam prediction with the top-K and bottom-K continuations (Go implementation)",
	}}
}

// kinds is the list for the model this server actually runs.
func (s *Service) kinds() []map[string]any {
	return Kinds(s.units())
}

// units is what the running model counts in: "chars", or "words" under a word
// encoding.
func (s *Service) units() string {
	if s.model == nil {
		return radixnet.DefaultEncoding().UnitsName()
	}
	return s.model.Encoding().UnitsName()
}

// Backends mirrors the Python status's backend availability block.
func (s *Service) Backends() map[string]any {
	return map[string]any{"python": false, "torch": false, "cuda": false, "mps": false, "go": true, "default": "go"}
}

// Status is GET /api/status.
// replaySummary is the model's replay buffer at a glance - {size, texts,
// seen} - or nil when it keeps none (../../SPEC-SearchAndTraining.md §4).
func (s *Service) replaySummary() any {
	out, err := s.read(func(m *radixnet.Model) (any, error) {
		if m.Replay == nil {
			return nil, nil
		}
		return map[string]any{"size": m.Replay.Size, "texts": m.Replay.Len(), "seen": m.Replay.Seen}, nil
	})
	if err != nil {
		return nil
	}
	return out
}

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
	kind, label, units := "count", ModelLabel, s.units()
	stats["kind"] = kind
	stats["model_label"] = label
	stats["units"] = units
	stats["kinds"] = s.kinds()
	stats["replay"] = s.replaySummary()
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
	var attention radixnet.AttentionConfig
	out, err := s.read(func(m *radixnet.Model) (any, error) {
		attention = m.AttentionConfig()
		return m.G.WeightConfig(), nil
	})
	if err != nil {
		return nil, err
	}
	var modelPath any
	if s.modelPath != "" {
		modelPath, _ = filepath.Abs(s.modelPath)
	}
	active, label, units := "count", ModelLabel, s.units()
	return map[string]any{
		"kind": active, "label": label, "units": units, "kinds": s.kinds(), "model_path": modelPath,
		"paths": map[string]any{active: modelPath}, "in_memory": []string{active}, "weights": out,
		"attention": attention, "engine": "go",
	}, nil
}

// Attention is GET /api/model/attention: where inside a gram the model's
// corrections land (radixnet/attention.go).
func (s *Service) Attention() (map[string]any, error) {
	out, err := s.read(func(m *radixnet.Model) (any, error) {
		return map[string]any{"kind": m.Kind(), "attention": m.AttentionConfig()}, nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// ConfigureAttention is POST /api/model/attention: the band on (at blur) or off.
func (s *Service) ConfigureAttention(on *bool, blur *float64) (map[string]any, error) {
	out, err := s.mutate(func(m *radixnet.Model) (any, error) {
		cfg, err := m.ConfigureAttention(on, blur)
		if err != nil {
			return nil, badRequest("%v", err)
		}
		return map[string]any{"kind": m.Kind(), "attention": cfg, "stats": m.Stats()}, nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// AttentionPreview is POST /api/model/attention/preview: where one correction
// would land, gram by gram, under the writer rule and a band; nothing changes.
func (s *Service) AttentionPreview(wrong, right string, blur *float64) (map[string]any, error) {
	out, err := s.read(func(m *radixnet.Model) (any, error) {
		p, err := m.AttentionPreview(wrong, right, blur)
		if err != nil {
			return nil, badRequest("%v", err)
		}
		return map[string]any{
			"kind": m.Kind(), "attention": p.Attention, "blur": p.Blur, "weights": p.Weights,
			"changes": p.Changes, "wrong": p.Wrong, "right": p.Right,
		}, nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// Encoding is GET /api/encoding: how the active model reads text - the unit,
// the n of the n-gram, the stride and the sentinels.
//
// A choice, and "configurable" says so - but one made when a model is created
// and fixed for its life, because the graph's labels, its splits and merges and
// its saved file are all written in it: POST /api/reset is where it is chosen.
func (s *Service) Encoding() map[string]any {
	enc := s.model.Encoding()
	return map[string]any{
		"encoding": enc.String(), "unit": string(enc.Unit),
		"window": enc.N, "ngram": enc.N, "stride": enc.Stride, "overlap": enc.Overlap(),
		"start_label": radixnet.StartLabel, "end_label": radixnet.EndLabel, "back_label": radixnet.BackLabel,
		"think_label":  radixnet.ThinkLabel,
		"configurable": true,
		"note": fmt.Sprintf(
			"Text goes in as %s, and comes back out of the (possibly compressed) node labels along a "+
				"path. The encoding is fixed for a model's life - every label is written in it - so it is "+
				"chosen when a model is made: POST /api/reset with {\"encoding\": \"word:2:1\"}, or "+
				"{unit, ngram, stride}.", enc.Describe()),
	}
}

// EncodingPreview is POST /api/encoding/preview: one text through the encoder
// and back through both decoders, and through the graph's own labels.
func (s *Service) EncodingPreview(text string) (map[string]any, error) {
	enc := s.model.Encoding()
	windows := enc.Encode(text)
	if windows == nil {
		windows = []string{}
	}
	decoded := enc.DecodeGrams(windows)
	info := s.Encoding()
	info["text"] = text
	info["chars"] = enc.Len(text)
	info["windows"] = windows
	info["count"] = len(windows)
	info["decoded"] = decoded
	// what comes back is what the encoding can represent: the text itself under a
	// sliding character window, its words under a word encoding
	info["round_trip"] = decoded == enc.Normalize(text)
	info["kind"] = "count"
	out, err := s.read(func(m *radixnet.Model) (any, error) {
		g := m.G
		unknown := []string{}
		for _, w := range windows {
			if _, _, ok := g.Lookup(w); !ok {
				unknown = append(unknown, w)
			}
		}
		info["unknown_windows"] = unknown
		walked, ok := []int(nil), false
		if len(windows) > 0 {
			walked, ok = g.NodePath(windows)
		}
		if !ok {
			return map[string]any{
				"known": false, "reason": encodingReason(enc, len(windows), unknown),
				"labels": []string{}, "node_ids": []int{}, "decoded": "", "nodes": 0, "compressed": 0,
			}, nil
		}
		labels := make([]string, len(walked))
		real := make([]string, 0, len(walked))
		compressed := 0
		for i, n := range walked {
			labels[i] = g.Labels[n]
			if n != radixnet.Start && n != radixnet.End {
				real = append(real, labels[i])
				if enc.Len(labels[i]) > enc.N {
					compressed++
				}
			}
		}
		return map[string]any{
			"known": true, "reason": nil, "labels": labels, "node_ids": walked,
			"decoded": enc.DecodePath(real, 0, true), "nodes": len(real), "compressed": compressed,
		}, nil
	})
	if err != nil {
		return nil, err
	}
	info["path"] = out
	return info, nil
}

// encodingReason says why a text cannot be walked: windows never seen, or a
// text whose windows are all known that still does not run from START to END.
func encodingReason(enc radixnet.Encoding, windows int, unknown []string) string {
	if windows == 0 {
		return fmt.Sprintf("the text is shorter than one window (%d %ss)", enc.N, enc.Unit)
	}
	if len(unknown) > 0 {
		shown := unknown
		suffix := ""
		if len(shown) > 5 {
			shown, suffix = shown[:5], "..."
		}
		quoted := make([]string, len(shown))
		for i, w := range shown {
			quoted[i] = fmt.Sprintf("%q", w)
		}
		return fmt.Sprintf("%d of the %d windows have never been seen: %s%s",
			len(unknown), windows, strings.Join(quoted, ", "), suffix)
	}
	return "every window is known, but the structure cannot walk the whole text from START to END as it " +
		"stands - a missing edge, a step into the middle of a merged node, or a text that is only part of " +
		"one it was trained on (the walk has to reach the end of a text)"
}

// SelectKind is POST /api/model/select: this server runs the model it was
// started with - the count / reward model, or the word model over the same
// graph - and selecting the other one is starting the server again with it.
func (s *Service) SelectKind(kind string) (map[string]any, error) {
	active := "count"
	wanted := strings.ToLower(strings.TrimSpace(kind))
	if wanted != active {
		if wanted == "count" || wanted == "word" {
			// both are this same model, over characters or over words: which one a server runs is
			// settled when it starts, because it is the model it holds
			return nil, badRequest("this Go server runs the %s model; the %s model needs "+
				"`radixnet-count --kind %s serve`", active, wanted, wanted)
		}
		return nil, badRequest("the Go server runs the count / reward model only, over characters or over words "+
			"(kind %q is served by the Python server)", kind)
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

// Reset replaces the model with a fresh one, in the encoding given (the zero
// Encoding is the default one: character trigrams).
func (s *Service) Reset(seed *int64, kind string, opts map[string]float64, enc radixnet.Encoding) (map[string]any, error) {
	if kind != "" && strings.ToLower(kind) != "count" {
		return nil, badRequest("the Go server runs the count / reward model only (kind %q is served by the Python server)", kind)
	}
	if err := s.ensureIdle(); err != nil {
		return nil, err
	}
	g := radixnet.DefaultGraphOptions()
	g.Encoding = enc.WithDefaults()
	if err := g.Encoding.Validate(); err != nil {
		return nil, badRequest("%v", err)
	}
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
// Paths is the judged paths: what each step did in the context it was taken from.
func (s *Service) Paths(limit int) (map[string]any, error) {
	if limit < 0 {
		return nil, badRequest("'limit' must be >= 0 (got %d)", limit)
	}
	out, err := s.read(func(m *radixnet.Model) (any, error) {
		g := m.G
		rows := []map[string]any{}
		for _, row := range m.Paths(limit, -1) {
			parent := g.ParentOfEdge(row.Edge)
			child := -1
			if parent >= 0 {
				for _, t := range g.Children(parent) {
					if t.E == row.Edge {
						child = t.P
					}
				}
			}
			rows = append(rows, map[string]any{
				"prev": row.Prev, "edge": row.Edge, "seen": row.Seen, "correct": row.Correct,
				"incorrect": row.Incorrect, "correct_ratio": row.CorrectRatio, "seen_ratio": row.SeenRatio,
				"term": row.Term, "after": g.Label(row.Prev), "parent": parent,
				"parent_label": g.Label(parent), "child": child, "child_label": g.Label(child),
			})
		}
		totals := g.PathTotals()
		return map[string]any{
			"totals": map[string]any{
				"contexts": totals.Contexts, "judged": totals.Judged, "seen": totals.Seen,
				"correct": totals.Correct, "incorrect": totals.Incorrect,
			},
			"paths": rows, "limit": limit, "path_scale": g.WeightConfig().PathScale,
		}, nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// Words is GET /api/words: a word encoding's alphabet, most read first.
func (s *Service) Words(limit int) (map[string]any, error) {
	if limit < 0 {
		return nil, badRequest("'limit' must be >= 0 (got %d)", limit)
	}
	out, err := s.read(func(m *radixnet.Model) (any, error) {
		enc := m.Encoding()
		if enc.Unit != radixnet.Words {
			return nil, badRequest(
				"this model counts in %s, so it has no words to list; a word alphabet needs a word encoding "+
					"(--encoding word:%d:%d)", enc.UnitsName(), enc.N, enc.Stride)
		}
		rows := m.TopWords(limit)
		if rows == nil {
			rows = []radixnet.WordRow{}
		}
		return map[string]any{
			"words": rows, "limit": limit, "vocabulary": len(enc.Vocabulary(m.G.GramIndex())),
			"units": enc.UnitsName(), "encoding": enc.String(),
		}, nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// NodeRatios is each node against the nodes around it: its traffic and its
// reward, shared out over the previous and the next nodes.
func (s *Service) NodeRatios(limit int, node string) (map[string]any, error) {
	if limit < 0 {
		return nil, badRequest("'limit' must be >= 0 (got %d)", limit)
	}
	out, err := s.read(func(m *radixnet.Model) (any, error) {
		g := m.G
		wanted := -1
		if node != "" {
			key := node // a label is text on every encoding
			for i := 0; i < g.NumNodeIDs(); i++ {
				if g.Label(i) == key {
					wanted = i
					break
				}
			}
			if wanted < 0 {
				found, _, ok := g.Lookup(key)
				if !ok {
					return nil, notFound("no node labelled %q: give a node label, or one of its trigrams", node)
				}
				wanted = found
			}
		}
		totals := g.PathTotals()
		rows := g.NodeRatioRows(limit, wanted)
		if rows == nil {
			rows = []radixnet.NodeStats{}
		}
		return map[string]any{
			"nodes": rows, "limit": limit, "node": nodeQuery(node), "total_nodes": g.NumNodes(),
			"totals": map[string]any{
				"contexts": totals.Contexts, "judged": totals.Judged, "seen": totals.Seen,
				"correct": totals.Correct, "incorrect": totals.Incorrect,
			},
		}, nil
	})
	if err != nil {
		return nil, err
	}
	return out.(map[string]any), nil
}

// nodeQuery echoes the ?node= that was asked for - null when none was, as the
// Python server reports it.
func nodeQuery(node string) any {
	if node == "" {
		return nil
	}
	return node
}

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
			ca, cb := g.NodeCount(sorted[a]), g.NodeCount(sorted[b])
			if ca != cb {
				return cb.Less(ca)
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
		nodes = append(nodes, map[string]any{"id": i, "label": g.Labels[i], "count": g.Count[i],
			"count_resets": g.CountResets[i], "activation": 1.0, "z": 0.0, "a": 0.0, "b": 1.0 / 3.0, "h": 0.0, "k": 1.0})
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
				"count_resets": g.EdgeCountResets[cc.Edge],
				"prob":         math.Exp(-cc.Cost), "cost": cc.Cost, "reward": g.EdgeReward[cc.Edge],
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
		"total_traversals": g.TotalTraversals.Value, "total_traversals_resets": g.TotalTraversals.Resets,
		"window_traversals": g.WindowTraversals(), "window": g.WindowSize,
	}
}

// Predict / Generate / Score / Converse run under the read lock.
//
// Every answer they hand out goes through the guard: the positive model
// writes and the negative network, built from nothing but the tutor's
// failures, vetoes what it recognises (radixnet.Filter).  Each returns the
// guard's report alongside the answer - nil when nothing guarded it.

// guard is the pair on the way out, or nil when there is nothing to guard
// with: no negative network in memory and none saved beside the model, or one
// that has never been taught a failure and would veto nothing.  An empty
// negative network is never created here - an answer is not the place to
// bring one into being.  Call it with the model lock held; the output paths
// do.
//
// provenance overrides the server's own setting for this one answer (POST
// /api/negative/settings {"provenance": false} sets it for every answer):
// off, the vetoes still apply but the report says how many, not which or why.
func (s *Service) guard(provenance *bool) *radixnet.Filter {
	negative := s.negative
	if negative == nil {
		path := s.negativePath()
		if path == "" {
			return nil
		}
		if _, err := os.Stat(path); err != nil {
			return nil
		}
		var err error
		if negative, err = s.negativeModel(); err != nil { // loads it from that file and parks it
			return nil
		}
	}
	config := s.guardConfig
	if provenance != nil {
		config.Provenance = *provenance // this answer's own choice
	}
	pair, err := radixnet.NewFilter(s.model, negative, config)
	if err != nil || !pair.Ready() {
		return nil
	}
	return pair
}

// guardReport is what the guard did, for the caller to show: the vetoes, with
// the reason and the fragment behind each.  Without provenance
// (FilterConfig.Provenance off) the report is the counts alone - how many
// candidates were judged and how many vetoed - and neither the vetoes nor the
// verdicts are listed.
func guardReport(pair *radixnet.Filter, verdicts []*radixnet.FilterVerdict, extra map[string]any) map[string]any {
	rejected := []*radixnet.FilterVerdict{}
	for _, verdict := range verdicts {
		if verdict.Decision == "reject" {
			rejected = append(rejected, verdict)
		}
	}
	var out map[string]any
	if pair.Config.Provenance {
		out = map[string]any{
			"on": true, "vetoed": len(rejected), "rejected": rejected, "verdicts": verdicts,
			"negative": pair.Negative.Stats(), "config": pair.Describe()["config"],
		}
	} else {
		out = map[string]any{
			"on": true, "provenance": false, "judged": len(verdicts), "vetoed": len(rejected),
			"negative": pair.Negative.Stats(), "config": pair.Describe()["config"],
		}
	}
	for k, v := range extra {
		out[k] = v
	}
	return out
}

// Predict continues a prefix; with guard the negative network vetoes the
// continuations it recognises as failures, and provenance (nil: the server's
// setting) says whether the report lists the vetoes or only counts them.
func (s *Service) Predict(prefix string, o radixnet.PredictOptions, guard bool, provenance *bool) (*radixnet.Prediction, map[string]any, error) {
	var report map[string]any
	out, err := s.read(func(m *radixnet.Model) (any, error) {
		found, err := m.Predict(prefix, o)
		if err != nil || !guard {
			return found, err
		}
		if pair := s.guard(provenance); pair != nil {
			ranked, verdicts := pair.Rank(prefix, found) // the survivors, best first
			kept := 0
			for _, verdict := range verdicts {
				if verdict.Decision != "reject" {
					kept++
				}
			}
			report = guardReport(pair, verdicts, map[string]any{"candidates": len(verdicts), "kept": kept})
			return ranked, nil
		}
		return found, nil
	})
	if err != nil {
		return nil, nil, badRequest("%v", err)
	}
	return out.(*radixnet.Prediction), report, nil
}

// Generate writes whole texts; with guard the model over-samples and the
// negative network vetoes what it recognises as failure (provenance as in
// Predict).
func (s *Service) Generate(o radixnet.GenerateOptions, guard bool, provenance *bool) ([]*radixnet.PathResult, map[string]any, error) {
	var report map[string]any
	out, err := s.read(func(m *radixnet.Model) (any, error) {
		pair := (*radixnet.Filter)(nil)
		if guard {
			pair = s.guard(provenance)
		}
		if pair == nil {
			return m.Generate(o)
		}
		count := o.Count
		outcome, err := pair.Generate(count, o)
		if err != nil {
			return nil, err
		}
		report = guardReport(pair, outcome.Verdicts, map[string]any{
			"candidates": outcome.Candidates, "kept": len(outcome.Kept), "asked": outcome.Asked, "rate": outcome.Rate,
		})
		return outcome.Results, nil
	})
	if err != nil {
		return nil, nil, badRequest("%v", err)
	}
	return out.([]*radixnet.PathResult), report, nil
}

func (s *Service) Score(text string) radixnet.Score {
	out, _ := s.read(func(m *radixnet.Model) (any, error) { return m.Score(text), nil })
	return out.(radixnet.Score)
}

// Think is POST /api/think: the active model thinks - one thought from the
// Think sentinel (Model.Think).  With o.Learn (the default) the thought teaches
// the model where it stopped to think, so - like a conversation - a thought
// changes the model; the server keeps that in memory until something saves.
// It takes the write lock and waits for it, as Python's session does: a running
// job hands the lock over between epochs, so a thought never has to be refused.
func (s *Service) Think(o radixnet.ThinkOptions) (map[string]any, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	thought, err := s.model.Think(o)
	if err != nil {
		return nil, badRequest("%v", err)
	}
	doc := thought.ToDict()
	doc["kind"] = s.model.Kind()
	return doc, nil
}

// StartTrainThoughts starts a train job that teaches thoughts as thoughts
// (Model.ThinkOn: texts whose walk begins at the Think sentinel, and - with
// questions - where a thought questions itself); answers, when given, are
// trained as ordinary texts after them.
func (s *Service) StartTrainThoughts(thoughts, answers []string, epochs int, questions bool) (map[string]any, error) {
	if epochs < 0 {
		return nil, badRequest("epochs must be >= 0, got %d", epochs)
	}
	if s.model.IsNegative() {
		return nil, badRequest("the negative network judges; it does not think")
	}
	return s.startJob("train", func(job *Job, progress func(map[string]any), stop func() bool) error {
		opts := radixnet.TrainOptions{Epochs: epochs, AutoCompress: true, Progress: progress, Stop: stop}
		if _, err := s.model.ThinkOn(thoughts, opts, questions, 1.0); err != nil {
			return err
		}
		if len(answers) > 0 && !stop() {
			if _, err := s.model.Train(answers, opts); err != nil {
				return err
			}
		}
		return nil
	})
}

// Converse holds a conversation; with guard a reply the negative network
// vetoes is left unsaid (provenance as in Predict).
func (s *Service) Converse(opening string, o radixnet.ConverseOptions, guard bool, provenance *bool) ([]*radixnet.Turn, map[string]any, error) {
	var report map[string]any
	out, err := s.read(func(m *radixnet.Model) (any, error) {
		pair := (*radixnet.Filter)(nil)
		if guard {
			pair = s.guard(provenance)
		}
		if pair == nil {
			return m.Converse(opening, o)
		}
		outcome, err := pair.Converse(opening, o)
		if err != nil {
			return nil, err
		}
		// "vetoed" counts the distinct texts refused, "refusals" how often one was (a turn may be
		// offered the same candidate again after its context was shortened)
		report = guardReport(pair, outcome.Verdicts, map[string]any{"refusals": outcome.Vetoed})
		return outcome.Turns, nil
	})
	if err != nil {
		return nil, nil, badRequest("%v", err)
	}
	return out.([]*radixnet.Turn), report, nil
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
