"""Right-to-left shaping that does not reverse the Latin inside a line.

``ui.pygame_display.rtl`` reverses the WHOLE string, which is correct for a
line that is entirely Hebrew and wrong the moment a line mixes scripts: it
renders ``English`` backwards. Its own docstring says so. Every label on the
existing protocol screens is pure Hebrew, so that has never shown -- but the
desk screens carry ``Enter``, ``Escape``, page counts and letter previews on
the same line as Hebrew.

This is not a bidirectional algorithm. It is the one rule those lines need:

* split the line into runs -- Hebrew, Latin/digit, and neutral;
* lay the runs out from the right, which rendered left-to-right means
  emitting them in reverse order;
* reverse the characters inside Hebrew runs, because the renderer draws
  left-to-right and Hebrew reads the other way;
* leave Latin and digit runs alone, which is the whole point;
* mirror the paired punctuation that has a mirror.

For a line that is entirely Hebrew this produces exactly what the existing
``rtl`` produces, which is what lets the two live side by side until the old
screens are converted. ``test_text_shaping`` pins that equivalence.
"""

from __future__ import annotations

# Hebrew, and the presentation block a few fonts still use.
_HEBREW_RANGES = ((0x0590, 0x05FF), (0xFB1D, 0xFB4F))
# Characters that belong to a Latin run rather than to the neutral space
# between runs: letters, digits, and the marks that sit inside a word.
_STRONG_LTR_EXTRA = "@#%&"
# Brackets and quotes swap hands in a right-to-left line.
_MIRROR = {
    "(": ")", ")": "(",
    "[": "]", "]": "[",
    "{": "}", "}": "{",
    "<": ">", ">": "<",
}

_HEB = "heb"
_LTR = "ltr"
_NEUTRAL = "neutral"


def is_hebrew_char(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _HEBREW_RANGES)


def has_hebrew(text: str) -> bool:
    return any(is_hebrew_char(ch) for ch in text)


def _kind(ch: str) -> str:
    if is_hebrew_char(ch):
        return _HEB
    if ch.isalnum() or ch in _STRONG_LTR_EXTRA:
        return _LTR
    return _NEUTRAL


def runs(text: str) -> list[tuple[str, str]]:
    """The line split into (kind, characters), in logical order."""

    out: list[tuple[str, str]] = []
    for ch in text:
        kind = _kind(ch)
        if out and out[-1][0] == kind:
            out[-1] = (kind, out[-1][1] + ch)
        else:
            out.append((kind, ch))
    return out


def _mirrored(text: str) -> str:
    return "".join(_MIRROR.get(ch, ch) for ch in text)


def shape_rtl(text: str) -> str:
    """Lay one line out right-to-left for a left-to-right renderer.

    A line with no Hebrew in it is returned untouched, so an English-only
    label is never disturbed.
    """

    if not has_hebrew(text):
        return text
    parts = []
    for kind, chunk in reversed(runs(text)):
        if kind == _LTR:
            parts.append(chunk)
        else:
            # Hebrew and neutral both reverse; neutral also swaps its
            # brackets, which would otherwise point the wrong way.
            parts.append(_mirrored(chunk)[::-1])
    return "".join(parts)
