# AudioImage — run commands.  `make help` lists every target.
#
# Everything here is the zero-dependency Python package driven through its
# command line (python3 only).  numpy is picked up automatically when it is
# installed and makes decoding roughly ten times faster; nothing requires it.
# Override any variable on the command line, e.g. `make demo RATE=44100`.

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---- settings ---------------------------------------------------------------
PY         ?= python3
AUDIOIMAGE ?= $(PY) -m audioimage
OUT        ?= out
DATA       ?= data
WAV        ?= $(OUT)/tone.wav
PNG        ?= $(OUT)/tone.png
BACK       ?= $(OUT)/tone-decoded.wav
KIND       ?= melody
NOTES      ?= C4 E4 G4 C5 G4 E4 C4
DURATION   ?= 2.5
RATE       ?= 22050
FREQ       ?= 440
N_FFT      ?= 1024
HOP        ?= 256
WINDOW     ?= hann
SCALE      ?= db
TOP_DB     ?= 80
FREQ_SCALE ?= linear
COLORMAP   ?= gray
VIEW_MAP   ?= magma
DEPTH      ?= 16
ITERS      ?= 64
BACKEND    ?= auto
FLAGS      ?=

ENCODE_FLAGS := --n-fft $(N_FFT) --hop $(HOP) --window $(WINDOW) --scale $(SCALE) \
                --top-db $(TOP_DB) --freq-scale $(FREQ_SCALE) --colormap $(COLORMAP) --depth $(DEPTH)

.PHONY: help tone encode decode roundtrip view waveform info demo test test-python \
        lint clean sample backends

help: ## List every target
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(OUT):
	@mkdir -p $(OUT)

tone: | $(OUT) ## Synthesise a test clip ($(KIND)) into $(WAV)
	$(AUDIOIMAGE) tone -o $(WAV) --kind $(KIND) --notes "$(NOTES)" --duration $(DURATION) \
		--rate $(RATE) --freq $(FREQ) $(FLAGS)

encode: | $(OUT) ## Encode $(WAV) into the picture $(PNG)
	$(AUDIOIMAGE) --backend $(BACKEND) encode $(WAV) -o $(PNG) $(ENCODE_FLAGS) $(FLAGS)

decode: | $(OUT) ## Decode $(PNG) back into $(BACK)
	$(AUDIOIMAGE) --backend $(BACKEND) decode $(PNG) -o $(BACK) --iters $(ITERS) $(FLAGS)

roundtrip: | $(OUT) ## Encode, decode and report what survived
	$(AUDIOIMAGE) --backend $(BACKEND) roundtrip $(WAV) --image $(PNG) -o $(BACK) \
		$(ENCODE_FLAGS) --iters $(ITERS) $(FLAGS)

view: | $(OUT) ## Draw a labelled chart of $(WAV) into $(OUT)/view.png
	$(AUDIOIMAGE) view $(WAV) -o $(OUT)/view.png --view-colormap $(VIEW_MAP) \
		--n-fft $(N_FFT) --hop $(HOP) --freq-scale $(FREQ_SCALE) $(FLAGS)

waveform: | $(OUT) ## Draw the samples of $(WAV) into $(OUT)/waveform.png
	$(AUDIOIMAGE) waveform $(WAV) -o $(OUT)/waveform.png $(FLAGS)

info: ## Describe $(PNG)
	$(AUDIOIMAGE) info $(PNG)

backends: ## Show which transform backend is in use
	@$(PY) -c "import json; from audioimage import describe_backends; print(json.dumps(describe_backends(), indent=2))"

demo: | $(OUT) ## The whole tour: synthesise, encode, view, decode, measure
	@echo "== 1. a clip to look at =="
	$(AUDIOIMAGE) tone -o $(WAV) --kind $(KIND) --notes "$(NOTES)" --duration $(DURATION) --rate $(RATE)
	@echo
	@echo "== 2. the waveform, and the plane it becomes =="
	$(AUDIOIMAGE) waveform $(WAV) -o $(OUT)/waveform.png
	$(AUDIOIMAGE) view $(WAV) -o $(OUT)/view.png --view-colormap $(VIEW_MAP) --n-fft $(N_FFT) --hop $(HOP)
	@echo
	@echo "== 3. encode, decode, and measure the round trip =="
	$(AUDIOIMAGE) --backend $(BACKEND) roundtrip $(WAV) --image $(PNG) -o $(BACK) $(ENCODE_FLAGS) --iters $(ITERS)
	@echo
	@echo "== 4. what the decoded audio looks like when it is analysed again =="
	$(AUDIOIMAGE) view $(BACK) -o $(OUT)/view-decoded.png --view-colormap $(VIEW_MAP) --n-fft $(N_FFT) --hop $(HOP)
	@echo
	@ls -la $(OUT)

sample: | $(OUT) ## Rebuild the sample clip committed in data/
	$(AUDIOIMAGE) tone -o $(DATA)/sample.wav --kind melody --notes "C4 E4 G4 C5 G4 E4 C4" \
		--duration 2.0 --rate 22050

test: ## Run the unit test suite
	$(PY) -m unittest discover -s tests -v

test-python: ## Run the suite on the standard-library backend, even where numpy exists
	AUDIOIMAGE_BACKEND=python $(PY) -m unittest discover -s tests

lint: ## Byte-compile every module (a syntax check without extra tools)
	$(PY) -m compileall -q audioimage tests && echo "ok"

clean: ## Remove generated files
	rm -rf $(OUT) __pycache__ */__pycache__ .pytest_cache build dist *.egg-info
