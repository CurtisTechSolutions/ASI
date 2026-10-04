# modelkit

The Python package: everything around a model — how it is taught, talked to,
served and seen. It was RadixCyclicNN's (`../../RadixCyclicNN/radixnet`), and
is a package of its own so other directories and repositories can use it.

**Standard library only.** Pillow, diffusers and a speech-to-text package are
optional extras for images and speech (`modelkit[images]`, `[diffusion]`,
`[speech]`, `[whisper]`).

Two layers, held apart by `../tests/test_layers.py`: the **model-free** modules
load with no model installed; the **model layer** is built on the `radixnet`
package (installed, or the RadixCyclicNN checkout beside this one). `../README.md`
says how to use either from another project.

## Talking to the outside (model-free)

| module | what it is |
|---|---|
| `llm.py` | provider-independent plumbing for the language models the network talks to, and `reader_text`: what a reader is shown of a text a model wrote (the words a model of sounds spells) |
| `ollama.py` | a local Ollama server: training corpora from a prompt, and adversarial review |
| `chatgpt.py` | ChatGPT over the hosted API, behind the same client interface |
| `agent_config.py` | load an `agent.json` chat connection and send Chat Completions requests; the model-free client behind `talk --agent-config` |
| `tools.py` | the external tools the network can call, and the text format it learns them in |
| `browser.py` | a real browser behind the web tools — Chrome over the W3C WebDriver protocol |
| `mcp.py` | an MCP server: the network's tools *and* the network itself, over the Model Context Protocol |
| `archive.py` | ZIP archives as training data: the text entries of an archive, unpacked safely |

## Other media (model-free)

| module | what it is |
|---|---|
| `vision.py` | images as text — the Stable Diffusion image encoder (image generation run backwards) plus base64 |
| `speech.py` | speech as text — what was said *and* the waveform that said it, both behind one unique token |
| `media.py` | what the two share: repairing the base64 payload a network predicted |

## Teaching it

| module | layer | what it is |
|---|---|---|
| `tutor.py` | model | automated English lessons: the tutor sets the exercise, the network completes it, the tutor marks it (and, thinking, teaches how it reasoned) |
| `critic.py` | model-free | the negative network feeding itself — an LLM reviewer on a loop |
| `recall.py` | model-free | what the network remembers of what it was shown — the recall tutor for speech and images |
| `chat.py` | model-free | the model in conversation with an LLM, and the LLM marking the conversation |
| `codegen.py` | model-free | code generation with a sandbox, an LLM teacher / judge and 2NRL rewards |
| `agent.py` | model | tool use: the network browses and solves on its own, an LLM sets the bar and teaches |
| `blame.py` | model-free | where the negative network's data comes from: the tutor's verdicts, turned into blame |
| `gan.py` | model | `Evolver` — the GAN-style self-upgrading loop: train on failures, fail on purpose, then invert |

A model-free loop is *handed* the model it teaches rather than importing one.

## Talking with it (the model layer)

| module | what it is |
|---|---|
| `dialogue.py` | the model conversing with itself (`converse(model, ...)`), with stutter detection and backtracking - and the stream of events a conversation is watched through as it happens |
| `thinking.py` | thoughts: walks that begin at the THINK sentinel, and the questions they ask themselves |
| `assistant.py` | today's format: the model answering in OpenAI's and Anthropic's chat dialects |
| `duo.py` | the two networks as one output path: the positive model writes, the negative one vetoes |
| `voice.py` | the model heard as it walks, and the output decoder that speaks any text in its units |
| `voicechat.py` | talking with the model by voice: heard, learned, answered and spoken |

## Plumbing (the model layer)

| module | what it is |
|---|---|
| `cli.py` | `python -m modelkit <command>` / `python -m radixnet` / `radixnet` — every command |
| `api.py` | the HTTP JSON API and the static file serving of `../frontend/dist`, standard library only |
| `checkpoint.py` | checkpoint directory management — rotation, the `latest` pointer, resume |
| `bench.py` | throughput benchmarks for the model |
| `__main__.py` | dispatches `python -m modelkit` to the CLI |
| `__init__.py` | the public surface, imported lazily so the model-free layer stays free |
