"""
engine/arabic_text.py

Arabic text utilities used when ingesting evidence and when routing a chat
question: language detection (which corpus, prompt and report language to
use) and repair of the lam-alef ligature defect that PDF text extraction
introduces.

The ligature defect
-------------------
Arabic renders lam + alef as a single ligature glyph (لا). Many PDF text
extractors emit that glyph's components in VISUAL order instead of logical
order, so the two letters come out swapped:

    correct  الأمن  =  ا + ل + أ + م + ن
    extracted األمن  =  ا + أ + ل + م + ن      <- lam and hamza-alef swapped

The bundled SAMA Arabic corpus had this defect throughout (1096 broken
trigrams and zero correct ones). It has since been repaired in place, so
ingestion/controls_ar.jsonl is now clean -- but the repair still runs on
EVIDENCE the user uploads, because Arabic PDFs supplied by a customer are
extracted at runtime and commonly carry the same defect. Left unrepaired it
degrades retrieval (the embedding model sees misspelled words) and looks
unprofessional in the generated report.

repair_lam_alef() targets only the unambiguous trigram "alef + alef-variant +
lam", which is the definite article ("ال") followed by a word starting with an
alef. That sequence does not occur in correct Arabic, so the repair cannot
corrupt clean text -- importantly it leaves genuine alef-lam words such as
"إلى" and "إلكتروني" untouched, which a naive two-letter swap would destroy.
"""
import re

# Arabic block, excluding Arabic-Indic digits so numeric-heavy text isn't
# mistaken for Arabic prose.
_ARABIC_LETTER_RE = re.compile(r"[ء-غف-ي]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")

# Bidi control characters that pdftotext leaves around embedded LTR numbers.
_BIDI_CONTROL_RE = re.compile(r"[‪-‮‎‏]")

# "alef + (alef-variant) + lam"  ->  "alef + lam + (alef-variant)"
_LAM_ALEF_RE = re.compile(r"ا([أإآا])ل")

# A standalone "ال" is never a valid Arabic word (the definite article is
# always attached to its noun); it is the negation particle "لا" flipped.
_STANDALONE_AL_RE = re.compile(r"(?<!\S)ال(?!\S)")


def repair_lam_alef(text: str) -> str:
    """Undo the lam-alef ligature swap left by PDF text extraction."""
    text = _LAM_ALEF_RE.sub(r"ال\1", text)
    text = _STANDALONE_AL_RE.sub("لا", text)
    return text


def strip_bidi_controls(text: str) -> str:
    return _BIDI_CONTROL_RE.sub("", text)


def clean_arabic(text: str) -> str:
    """Full clean-up for Arabic text coming out of a PDF/DOCX extractor."""
    return repair_lam_alef(strip_bidi_controls(text))


def detect_language(text: str, threshold: float = 0.15) -> str:
    """Return 'ar' when the text is predominantly Arabic, else 'en'.

    Compares Arabic vs Latin letter counts rather than Arabic-vs-everything,
    so numbers, punctuation and the odd English acronym inside an Arabic
    policy (VPN, MFA, SAMA) don't flip the verdict. The threshold is a share
    of *letters*, deliberately low because Arabic technical documents quote a
    lot of English terminology.
    """
    if not text:
        return "en"
    arabic = len(_ARABIC_LETTER_RE.findall(text))
    latin = len(_LATIN_LETTER_RE.findall(text))
    total = arabic + latin
    if total == 0:
        return "en"
    return "ar" if (arabic / total) >= threshold else "en"
