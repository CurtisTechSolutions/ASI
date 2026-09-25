"""Numbers as words: what a digit sounds like is the word for it.

``42`` is *forty two*, ``3.5`` *three point five*, ``1,000`` *one thousand*,
``21st`` *twenty first*, ``-7`` *minus seven*, ``50%`` *fifty percent*,
``$5`` *five dollars*.  A token that mixes letters and digits (``mp3``,
``3d``, ``b2b``) is split into its runs, so the rest of the tokeniser sees
words and letters.  Everything here is deterministic and covers what prose
holds; it is not a locale library.
"""

from __future__ import annotations

import re

ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
        "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
SCALES = ("", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion", "sextillion",
          "septillion", "octillion", "nonillion", "decillion")
ORDINALS = {
    "one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
    "twelve": "twelfth",
}


def cardinal(n: int) -> str:
    """``1234 -> "one thousand two hundred thirty four"``; negative numbers say *minus*."""
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return TENS[tens] + (" " + ONES[ones] if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return ONES[hundreds] + " hundred" + (" " + cardinal(rest) if rest else "")
    parts: list[str] = []
    scale = 0
    while n:
        n, chunk = divmod(n, 1000)
        if chunk:
            name = SCALES[scale] if scale < len(SCALES) else f"ten to the {scale * 3}"
            parts.append(cardinal(chunk) + (" " + name if name else ""))
        scale += 1
    return " ".join(reversed(parts))


def ordinal(n: int) -> str:
    """``21 -> "twenty first"``, ``100 -> "one hundredth"``."""
    words = cardinal(n).split()
    last = words[-1]
    if last in ORDINALS:
        words[-1] = ORDINALS[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def digits(text: str) -> str:
    """Each digit by name: ``"007" -> "zero zero seven"``."""
    return " ".join(ONES[int(ch)] for ch in text if ch.isdigit())


_NUMBER = re.compile(
    r"""
    (?P<currency>[$€£])?
    (?P<sign>[-+−])?
    (?P<int>\d{1,3}(?:,\d{3})+|\d+)
    (?:\.(?P<frac>\d+))?
    (?P<ordinal>st|nd|rd|th)?
    (?P<percent>%)?
    """,
    re.VERBOSE | re.IGNORECASE,
)

_CURRENCY = {"$": ("dollar", "dollars"), "€": ("euro", "euros"), "£": ("pound", "pounds")}


def number_words(token: str) -> str | None:
    """The words of a number token, or ``None`` if it is not one.

    ``"42"``, ``"3.14"``, ``"1,000"``, ``"-7"``, ``"21st"``, ``"50%"``, ``"$5"``, ``"$2.50"``
    are numbers; ``"3d"`` and ``"mp3"`` are not (see :func:`split_alphanumeric`).
    """
    m = _NUMBER.fullmatch(token)
    if not m:
        return None
    whole = int(m.group("int").replace(",", ""))
    words: list[str] = []
    if m.group("sign") in ("-", "−"):
        words.append("minus")
    frac = m.group("frac")
    if m.group("ordinal"):
        if frac:
            return None
        words.append(ordinal(whole))
    elif m.group("currency") and frac and len(frac) == 2:
        unit = _CURRENCY[m.group("currency")]
        words.append(cardinal(whole))
        words.append(unit[0] if whole == 1 else unit[1])
        cents = int(frac)
        if cents:
            words.append(cardinal(cents))
            words.append("cent" if cents == 1 else "cents")
        return " ".join(words)
    else:
        words.append(cardinal(whole))
        if frac is not None:
            words.append("point")
            words.append(digits(frac))
    if m.group("currency"):
        unit = _CURRENCY[m.group("currency")]
        words.append(unit[0] if whole == 1 and not frac else unit[1])
    if m.group("percent"):
        words.append("percent")
    return " ".join(words)


_RUNS = re.compile(r"\d+|[^\d]+")


def split_alphanumeric(token: str) -> list[str]:
    """``"mp3" -> ["mp", "3"]``, ``"3d" -> ["3", "d"]``; a token without digits comes back alone."""
    if not any(ch.isdigit() for ch in token):
        return [token]
    return _RUNS.findall(token)
