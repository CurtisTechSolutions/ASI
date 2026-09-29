"""``python -m radixnet`` - run the command line, which lives in the ``modelkit`` package.

The model is this package; the command line built around it is
``modelkit.cli`` (``../ModelKit``).  When ``modelkit`` is not installed, the
checkout beside this one is used, the way :func:`radixnet.encoding.phonetok_module`
finds the phonetic tokenizer.  This entry point is the one place the model
package reaches for the kit: nothing the model is made of imports it.
"""

import os
import sys


def main(argv=None) -> int:
    try:
        from modelkit.cli import main as cli_main
    except ModuleNotFoundError as exc:
        if exc.name != "modelkit":
            raise
        sibling = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ModelKit")
        if not os.path.isdir(os.path.join(sibling, "modelkit")):
            raise SystemExit(
                "the radixnet command line is the modelkit package: install it (pip install ../ModelKit) "
                "or keep the ModelKit checkout beside this one"
            ) from exc
        sys.path.append(sibling)
        from modelkit.cli import main as cli_main
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
