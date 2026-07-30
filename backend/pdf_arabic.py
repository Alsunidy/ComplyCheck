"""
ComplyCheck - Arabic support for the ReportLab PDF report.

Rendering Arabic in a PDF needs three things that Latin text does not:

  1. A font containing Arabic glyphs. ReportLab's built-in Type1 faces
     (Helvetica/Times/Courier) have none, so Arabic would come out as blank
     boxes -- a TrueType font must be registered explicitly.
  2. Letter shaping. Arabic letters change form depending on their position
     in a word (initial/medial/final/isolated). arabic_reshaper converts the
     logical characters into the correct presentation forms.
  3. Bidirectional reordering. PDF draws glyphs left-to-right in the order
     given, so right-to-left text must be reversed by the Unicode bidi
     algorithm first -- which also keeps embedded LTR runs (control IDs like
     "3.1.1", percentages, filenames) in their correct order.

Font resolution order: COMPLYCHECK_ARABIC_FONT env var, then a bundled font
under backend/assets/fonts/, then the usual system locations on Windows,
macOS and Linux.
"""
import os
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ARABIC_FONT_NAME = "ComplyCheckArabic"
ARABIC_FONT_BOLD_NAME = "ComplyCheckArabic-Bold"

_ASSETS_FONTS = Path(__file__).resolve().parent / "assets" / "fonts"

# (regular, bold) candidates, most preferred first.
_FONT_CANDIDATES = [
    (_ASSETS_FONTS / "NotoNaskhArabic-Regular.ttf", _ASSETS_FONTS / "NotoNaskhArabic-Bold.ttf"),
    (_ASSETS_FONTS / "Amiri-Regular.ttf", _ASSETS_FONTS / "Amiri-Bold.ttf"),
    (Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\arialbd.ttf")),
    (Path(r"C:\Windows\Fonts\tahoma.ttf"), Path(r"C:\Windows\Fonts\tahomabd.ttf")),
    (Path(r"C:\Windows\Fonts\segoeui.ttf"), Path(r"C:\Windows\Fonts\segoeuib.ttf")),
    (Path("/Library/Fonts/Arial Unicode.ttf"), Path("/Library/Fonts/Arial Unicode.ttf")),
    (Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
     Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")),
    (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
     Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    (Path("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"),
     Path("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf")),
]

_registered = False


class ArabicFontMissing(RuntimeError):
    """Raised when no Arabic-capable TrueType font can be found."""


def _candidate_pairs():
    override = os.environ.get("COMPLYCHECK_ARABIC_FONT")
    if override:
        bold = os.environ.get("COMPLYCHECK_ARABIC_FONT_BOLD", override)
        yield Path(override), Path(bold)
    yield from _FONT_CANDIDATES


def ensure_arabic_font() -> str:
    """Register an Arabic-capable font with ReportLab and return its name."""
    global _registered
    if _registered:
        return ARABIC_FONT_NAME

    for regular, bold in _candidate_pairs():
        if not regular.exists():
            continue
        pdfmetrics.registerFont(TTFont(ARABIC_FONT_NAME, str(regular)))
        bold_path = bold if bold.exists() else regular
        pdfmetrics.registerFont(TTFont(ARABIC_FONT_BOLD_NAME, str(bold_path)))
        pdfmetrics.registerFontFamily(
            ARABIC_FONT_NAME, normal=ARABIC_FONT_NAME, bold=ARABIC_FONT_BOLD_NAME
        )
        _registered = True
        return ARABIC_FONT_NAME

    raise ArabicFontMissing(
        "No Arabic-capable TrueType font was found. Set COMPLYCHECK_ARABIC_FONT "
        "to a .ttf path, or place NotoNaskhArabic-Regular.ttf in "
        f"{_ASSETS_FONTS}."
    )


def shape(text: str) -> str:
    """Shape + bidi-reorder Arabic text for correct PDF rendering.

    Safe to call on any string: Latin-only text passes through effectively
    unchanged.
    """
    if not text:
        return text
    import arabic_reshaper
    from bidi.algorithm import get_display

    # ReportLab markup (<b>, <br/>) and HTML entities (&mdash;) must survive
    # untouched: reshaping them turns "&mdash;" into ";mdash&" and bidi
    # reordering scrambles tag syntax.
    import re

    parts = re.split(r"(<[^>]+>|&[a-zA-Z]+;|&#\d+;)", text)
    if len(parts) > 1:
        return "".join(
            part if part.startswith("<") or part.startswith("&")
            else get_display(arabic_reshaper.reshape(part))
            for part in parts
        )
    return get_display(arabic_reshaper.reshape(text))


def shape_wrapped(text: str, font_name: str, font_size: float, max_width: float) -> str:
    """Shape Arabic text and pre-wrap it into correctly ordered lines.

    Why this exists: get_display() reverses a whole paragraph at once. If
    ReportLab then wraps that reversed string, the LINES come out in reverse
    order too -- the closing words of a sentence appear on the first line.
    So the text is wrapped here first (measuring the shaped glyphs against
    the available width), each line is bidi-reordered independently, and the
    lines are joined with <br/> so ReportLab does no wrapping of its own.

    `max_width` is the usable width in points (column width minus padding).
    """
    if not text:
        return text
    import arabic_reshaper
    from bidi.algorithm import get_display
    from reportlab.pdfbase.pdfmetrics import stringWidth

    reshaped = arabic_reshaper.reshape(text)

    lines: list[str] = []
    for hard_line in reshaped.split("\n"):
        current = ""
        for word in hard_line.split(" "):
            candidate = f"{current} {word}".strip()
            if not current or stringWidth(candidate, font_name, font_size) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)

    # Reorder each line on its own, preserving top-to-bottom line order.
    return "<br/>".join(get_display(line) for line in lines if line != "" or True)
