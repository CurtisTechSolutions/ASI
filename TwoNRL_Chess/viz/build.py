"""Inject ``data.json`` into ``index.html``'s data block, in place.

The page is one self-contained file - the Artifact sandbox blocks a page from
fetching its own data at runtime, and a single file is also what makes the
visualisation openable straight out of a clone.  So the data goes inside it,
into the ``<script id="viz-data">`` block, and this script is the only thing
that writes there.  ``index.html`` stays directly editable: everything outside
that one block is left exactly as it was found.
"""

from __future__ import annotations

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK = re.compile(
    r'(<script id="viz-data" type="application/json">)(.*?)(</script>)',
    re.DOTALL,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--page", default=os.path.join(HERE, "index.html"))
    p.add_argument("--data", default=os.path.join(HERE, "data.json"))
    args = p.parse_args()

    with open(args.data) as fh:
        payload = json.load(fh)
    with open(args.page) as fh:
        html = fh.read()
    if not BLOCK.search(html):
        raise SystemExit(f'{args.page} has no <script id="viz-data"> block to write into')

    # </script> inside a JSON string would close the block early; < cannot.
    blob = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    out = BLOCK.sub(lambda m: m.group(1) + blob + m.group(3), html, count=1)
    with open(args.page, "w") as fh:
        fh.write(out)
    print(f"wrote {args.page}  ({len(out) / 1024:.0f} KB, data {len(blob) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
