"""The two layers of the kit, and the one direction the dependency runs.

* The model-free modules load with no model package at all: a child Python
  whose ``radixnet`` import is refused still imports every one of them.
* Nothing in the model package imports the kit: ``radixnet``'s modules never
  name ``modelkit``, except its ``python -m radixnet`` entry point, which hands
  over to the kit's command line and is the one sanctioned exception.
* ``import modelkit`` itself loads no model: its public names are lazy.
"""

import ast
import os
import subprocess
import sys
import textwrap
import unittest

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(os.path.dirname(KIT), "RadixCyclicNN", "radixnet")

MODEL_FREE = ("archive", "blame", "browser", "chat", "chatgpt", "codegen", "critic", "llm", "mcp", "media",
              "ollama", "recall", "speech", "tools", "vision")
"""The kit modules that need no model to load (see ``modelkit/__init__.py``)."""

MODEL_LAYER = ("agent", "api", "assistant", "bench", "checkpoint", "cli", "dialogue", "duo", "gan", "thinking",
               "tutor", "voice", "voicechat")
"""The kit modules built on the model package."""

_BLOCKED_CHILD = textwrap.dedent("""
    import importlib.abc, sys

    class Refuse(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "radixnet" or name.startswith("radixnet."):
                raise ModuleNotFoundError(f"No module named {name!r} (refused by the layer test)", name=name)
            return None

    sys.meta_path.insert(0, Refuse())
    sys.path.insert(0, sys.argv[1])
    import importlib
    import modelkit
    assert "radixnet" not in sys.modules, "import modelkit loaded the model"
    for name in sys.argv[2:]:
        importlib.import_module("modelkit." + name)
    loaded = sorted(m for m in sys.modules if m == "radixnet" or m.startswith("radixnet."))
    assert not loaded, loaded
    print("ok")
""")


class TestLayers(unittest.TestCase):
    def test_the_model_free_layer_loads_without_a_model(self):
        proc = subprocess.run([sys.executable, "-c", _BLOCKED_CHILD, KIT, *MODEL_FREE],
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "ok")

    def test_the_model_layer_is_the_rest(self):
        package = os.path.join(KIT, "modelkit")
        modules = {f[:-3] for f in os.listdir(package) if f.endswith(".py") and not f.startswith("__")}
        self.assertEqual(modules, set(MODEL_FREE) | set(MODEL_LAYER),
                         "a new kit module belongs to one of the two layers: add it to MODEL_FREE or MODEL_LAYER")
        self.assertFalse(set(MODEL_FREE) & set(MODEL_LAYER))

    def test_the_model_layer_needs_the_model(self):
        """Each model-layer module really does load the model: the split above is not stale."""
        for name in MODEL_LAYER:
            with self.subTest(module=name):
                proc = subprocess.run([sys.executable, "-c", _BLOCKED_CHILD, KIT, name],
                                      capture_output=True, text=True, timeout=120)
                self.assertNotEqual(proc.returncode, 0, f"modelkit.{name} loads without the model: move it to MODEL_FREE")

    @unittest.skipUnless(os.path.isdir(MODEL), "the RadixCyclicNN checkout is not beside this one")
    def test_the_model_never_imports_the_kit(self):
        for fname in sorted(os.listdir(MODEL)):
            if not fname.endswith(".py") or fname == "__main__.py":
                continue
            with self.subTest(module=fname):
                with open(os.path.join(MODEL, fname), encoding="utf-8") as fh:
                    tree = ast.parse(fh.read())
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [a.name for a in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        names = [node.module]
                    for name in names:
                        self.assertFalse(name == "modelkit" or name.startswith("modelkit."),
                                         f"radixnet/{fname} imports {name}")


if __name__ == "__main__":
    unittest.main()
