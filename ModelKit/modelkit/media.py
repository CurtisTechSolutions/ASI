"""What the media text formats share: repairing the base64 payload a network predicted.

The media encoders (:mod:`modelkit.vision`, :mod:`modelkit.speech`) pack their
payload as base64 into a text a network trains on and *predicts* (``img:...``,
``aud:...``), so what comes back is rarely clean: it is repaired before it is
decoded.  Standard library only, and nothing here knows about a model.
"""

from __future__ import annotations

import base64
import binascii
import re

__all__ = ["repair_base64"]

_B64_JUNK = re.compile(r"[^A-Za-z0-9+/=]")


def repair_base64(body: str) -> tuple[bytes, bool]:
    """Decode the base64 tail of a media text, repairing it first; ``(payload, repaired)``.

    The media encoders (:mod:`modelkit.vision`, :mod:`modelkit.speech`) pack
    their payload as base64 into a text the network trains on and *predicts*,
    so what comes back may be cut off, padded with junk or interrupted by
    whitespace.  Characters outside the base64 alphabet are dropped, a single
    dangling character (which can never decode) is removed with them, the
    padding is completed, and ``repaired`` says whether any of that changed
    the text.  The payload is returned as it decodes - callers pad or truncate
    it to the length their format needs.
    """
    clean = _B64_JUNK.sub("", body).rstrip("=")
    if len(clean) % 4 == 1:  # a single dangling character can never decode
        clean = clean[:-1]
    padded = clean + "=" * (-len(clean) % 4)
    repaired = padded != body.strip()  # a clean text comes back unchanged, padding included
    try:
        payload = base64.b64decode(padded, validate=True)
    except (ValueError, binascii.Error) as exc:  # pragma: no cover - the junk filter makes this rare
        raise ValueError(f"the base64 part cannot be decoded: {exc}") from exc
    return payload, repaired
