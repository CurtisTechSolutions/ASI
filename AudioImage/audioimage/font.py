"""A 5x7 bitmap font, so a picture can label its own axes.

Drawing "8 kHz" beside a row needs glyphs, and the alternatives - a font file,
a font library - would both be a dependency for the sake of a few hundred
pixels.  The classic 5x7 cell font is small enough to simply carry: 95 glyphs
for ASCII 32 to 126, five columns each, one byte per column with the top row in
bit 0, written below as hex.

    for column_bits in glyph("A"):
        for row in range(CHAR_HEIGHT):
            if column_bits >> row & 1:
                ...  # this pixel is ink
"""

from __future__ import annotations

__all__ = ["CHAR_HEIGHT", "CHAR_WIDTH", "FIRST_CHAR", "LAST_CHAR", "SPACING", "glyph", "text_width"]

CHAR_WIDTH = 5
CHAR_HEIGHT = 7
FIRST_CHAR = 32
LAST_CHAR = 126
SPACING = 1
"""Blank columns between glyphs, so a cell is effectively six wide."""

_TABLE = (
    "0000000000"  # space
    "00005f0000"  # !
    "0007000700"  # "
    "147f147f14"  # #
    "242a7f2a12"  # $
    "2313086462"  # %
    "3649552250"  # &
    "0005030000"  # '
    "001c224100"  # (
    "0041221c00"  # )
    "14083e0814"  # *
    "08083e0808"  # +
    "0050300000"  # ,
    "0808080808"  # -
    "0060600000"  # .
    "2010080402"  # /
    "3e5149453e"  # 0
    "00427f4000"  # 1
    "4261514946"  # 2
    "2141454b31"  # 3
    "1814127f10"  # 4
    "2745454539"  # 5
    "3c4a494930"  # 6
    "0171090503"  # 7
    "3649494936"  # 8
    "064949291e"  # 9
    "0036360000"  # :
    "0056360000"  # ;
    "0814224100"  # <
    "1414141414"  # =
    "0041221408"  # >
    "0201510906"  # ?
    "324979413e"  # @
    "7e1111117e"  # A
    "7f49494936"  # B
    "3e41414122"  # C
    "7f4141221c"  # D
    "7f49494941"  # E
    "7f09090901"  # F
    "3e4149497a"  # G
    "7f0808087f"  # H
    "00417f4100"  # I
    "2040413f01"  # J
    "7f08142241"  # K
    "7f40404040"  # L
    "7f020c027f"  # M
    "7f0408107f"  # N
    "3e4141413e"  # O
    "7f09090906"  # P
    "3e4151215e"  # Q
    "7f09192946"  # R
    "4649494931"  # S
    "01017f0101"  # T
    "3f4040403f"  # U
    "1f2040201f"  # V
    "3f4038403f"  # W
    "6314081463"  # X
    "0708700807"  # Y
    "6151494543"  # Z
    "007f414100"  # [
    "0204081020"  # backslash
    "0041417f00"  # ]
    "0402010204"  # ^
    "4040404040"  # _
    "0001020400"  # `
    "2054545478"  # a
    "7f48444438"  # b
    "3844444420"  # c
    "384444487f"  # d
    "3854545418"  # e
    "087e090102"  # f
    "0c5252523e"  # g
    "7f08040478"  # h
    "00447d4000"  # i
    "2040443d00"  # j
    "7f10284400"  # k
    "00417f4000"  # l
    "7c0418047c"  # m
    "7c08040478"  # n
    "3844444438"  # o
    "7c14141408"  # p
    "081414187c"  # q
    "7c08040408"  # r
    "4854545420"  # s
    "043f444020"  # t
    "3c4040207c"  # u
    "1c2040201c"  # v
    "3c4030403c"  # w
    "4428102844"  # x
    "0c5050503c"  # y
    "4464544c44"  # z
    "0008364100"  # {
    "00007f0000"  # |
    "0041360800"  # }
    "08082a1c08"  # ~
)

_GLYPHS: tuple[tuple[int, ...], ...] = tuple(
    tuple(int(_TABLE[i * 10 + c * 2 : i * 10 + c * 2 + 2], 16) for c in range(CHAR_WIDTH))
    for i in range(LAST_CHAR - FIRST_CHAR + 1)
)

if len(_TABLE) != (LAST_CHAR - FIRST_CHAR + 1) * CHAR_WIDTH * 2:  # pragma: no cover
    raise ValueError(f"the font table is {len(_TABLE)} characters, expected {(LAST_CHAR - FIRST_CHAR + 1) * 10}")


def glyph(char: str) -> tuple[int, ...]:
    """The five column bitmaps of ``char``; anything outside the table is blank."""
    code = ord(char[0]) if char else FIRST_CHAR
    if code < FIRST_CHAR or code > LAST_CHAR:
        return (0, 0, 0, 0, 0)
    return _GLYPHS[code - FIRST_CHAR]


def text_width(text: str, scale: int = 1) -> int:
    """How many pixels wide ``text`` will be when drawn at ``scale``."""
    if not text:
        return 0
    return (len(text) * (CHAR_WIDTH + SPACING) - SPACING) * scale
