"""modelkit - everything around a model: how it is taught, talked to, served and seen.

This is what grew around the RadixCyclicNN model (``../RadixCyclicNN``), broken
out so other directories and repositories can use it: the teaching loops, the
LLM clients, the agent and its tools, speech, images and voice, the Model
Context Protocol server, the command line, the HTTP API and the React frontend
(``../frontend``).  Pure Python, standard library only; the optional extras are
the ones the media need (``pip install modelkit[images]``, ``[speech]``, ...).

Two layers, and the line between them is enforced by a test
(``tests/test_layers.py``):

* **Model-free** - these load with no model installed at all: the LLM clients
  (:mod:`~modelkit.llm`, :mod:`~modelkit.ollama`, :mod:`~modelkit.chatgpt`),
  the agent's tools and browser (:mod:`~modelkit.tools`,
  :mod:`~modelkit.browser`), the MCP server over them (:mod:`~modelkit.mcp`),
  archive uploads (:mod:`~modelkit.archive`), audio and images as text
  (:mod:`~modelkit.speech`, :mod:`~modelkit.vision`, :mod:`~modelkit.media`),
  and the loops that are handed a model rather than importing one - the
  critic, code generation, the recall tutor, the LLM chat and blame
  (:mod:`~modelkit.critic`, :mod:`~modelkit.codegen`, :mod:`~modelkit.recall`,
  :mod:`~modelkit.chat`, :mod:`~modelkit.blame`).
* **Built on the model package** ``radixnet`` - the tutor, the agent, evolve,
  the conversation and thinking walks, the assistant formats, the negative
  filter, voice, checkpoints, the benchmark, the HTTP API and the command line
  (:mod:`~modelkit.tutor`, :mod:`~modelkit.agent`, :mod:`~modelkit.gan`,
  :mod:`~modelkit.dialogue`, :mod:`~modelkit.thinking`,
  :mod:`~modelkit.assistant`, :mod:`~modelkit.duo`, :mod:`~modelkit.voice`,
  :mod:`~modelkit.voicechat`, :mod:`~modelkit.checkpoint`,
  :mod:`~modelkit.bench`, :mod:`~modelkit.api`, :mod:`~modelkit.cli`).

The dependency runs one way: this package imports ``radixnet``, and nothing in
``radixnet`` imports this one (its ``python -m radixnet`` entry point hands
over to :mod:`modelkit.cli`, and that is all).  When ``radixnet`` is not
installed, the RadixCyclicNN checkout beside this one is used, the way the
model finds the phonetic tokenizer beside it.

Importing ``modelkit`` itself loads nothing but this file: the names below are
imported from their modules on first use.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys

__version__ = "0.1.0"


def _find_model_package() -> None:
    """Put the RadixCyclicNN checkout beside this one on the path when ``radixnet`` is not installed."""
    try:
        if importlib.util.find_spec("radixnet") is not None:
            return
    except ImportError:  # an import hook that refuses the model reads as a model that is not there
        pass
    sibling = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "RadixCyclicNN")
    if os.path.isdir(os.path.join(sibling, "radixnet")) and sibling not in sys.path:
        sys.path.append(sibling)


_find_model_package()

# name -> (module, attribute): the public names, imported on first use so the model-free layer stays model-free
_EXPORTS = {
    "Ask": ("assistant", "Ask"),
    "Reply": ("assistant", "Reply"),
    "complete": ("assistant", "complete"),
    "respond": ("assistant", "respond"),
    "stream": ("assistant", "stream"),
    "to_anthropic": ("assistant", "to_anthropic"),
    "to_openai": ("assistant", "to_openai"),
    "CheckpointManager": ("checkpoint", "CheckpointManager"),
    "EvolveConfig": ("gan", "EvolveConfig"),
    "Evolver": ("gan", "Evolver"),
    "FilterConfig": ("duo", "FilterConfig"),
    "NegativeFilter": ("duo", "NegativeFilter"),
    "converse": ("dialogue", "converse"),
    "ASR_BACKENDS": ("speech", "ASR_BACKENDS"),
    "SPEECH_CODECS": ("speech", "CODECS"),
    "SPEECH_RATE": ("speech", "DEFAULT_RATE"),
    "SPEECH_TOKEN": ("speech", "SPEECH_TOKEN"),
    "Audio": ("speech", "Audio"),
    "SpeechError": ("speech", "SpeechError"),
    "encode_audio": ("speech", "encode_audio"),
    "speech_texts": ("speech", "speech_texts"),
    "teach_by_speech": ("speech", "teach"),
    "transcribe": ("speech", "transcribe"),
    "utterance_token": ("speech", "utterance_token"),
    "LLMClient": ("llm", "LLMClient"),
    "LLMError": ("llm", "LLMError"),
    "make_client": ("llm", "make_client"),
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str):
    """Import a public name from its module the first time it is asked for (PEP 562)."""
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'modelkit' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f".{module}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
