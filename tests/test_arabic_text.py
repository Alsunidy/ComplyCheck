"""Arabic ligature repair and language detection.

These carry the highest regression risk in the project: a naive "swap alef
and lam" repair silently destroys common, correctly-spelled words, and a
language mis-detection routes the whole audit to the wrong corpus.
"""
import json
from pathlib import Path

import pytest

import arabic_text

CONTROLS_AR = (
    Path(__file__).resolve().parent.parent
    / "backend" / "engine" / "ingestion" / "controls_ar.jsonl"
)


class TestRepairLamAlef:
    @pytest.mark.parametrize(
        "corrupt, expected, why",
        [
            ("حوكمة األمن", "حوكمة الأمن", "definite article + hamza-alef word"),
            ("يجب االستناد", "يجب الاستناد", "definite article + plain-alef word"),
            ("مجلس اإلدارة", "مجلس الإدارة", "definite article + hamza-below word"),
            ("ال يجوز ذلك", "لا يجوز ذلك", "standalone ال is the negation particle"),
        ],
    )
    def test_repairs_flipped_ligatures(self, corrupt, expected, why):
        assert arabic_text.repair_lam_alef(corrupt) == expected, why

    @pytest.mark.parametrize(
        "text, why",
        [
            ("إلى المؤسسة المالية", "إلى is a real word, not a flipped ligature"),
            ("الخدمات الإلكترونية", "إلكتروني legitimately starts with إل"),
            ("النظام الآلي", "آلي legitimately starts with آل"),
            ("الحالات غير المألوفة", "مألوف legitimately contains أل"),
            ("الأمن السيبراني", "already-correct text must not change"),
            ("Latin text only", "non-Arabic passes through"),
            ("", "empty input is safe"),
        ],
    )
    def test_leaves_correct_text_untouched(self, text, why):
        assert arabic_text.repair_lam_alef(text) == text, why

    def test_is_idempotent(self):
        once = arabic_text.repair_lam_alef("حوكمة األمن السيبراني")
        assert arabic_text.repair_lam_alef(once) == once


class TestDetectLanguage:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("The organization shall establish a policy", "en"),
            ("يجب على المؤسسة المالية تحديد سياسة", "ar"),
            # Arabic policies quote English acronyms constantly; that must not
            # flip the detected language.
            ("يجب تفعيل MFA على جميع اتصالات VPN", "ar"),
            ("Enable MFA for all VPN connections", "en"),
            ("", "en"),
            ("12345 -- 67890", "en"),
        ],
    )
    def test_detects(self, text, expected):
        assert arabic_text.detect_language(text) == expected


class TestArabicCorpusIsClean:
    """The shipped Arabic corpus must contain no known extraction defects.

    This guards the data itself: the corpus was repaired in place, and a
    re-import from the source PDF could silently reintroduce the defect.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def corpus_text(cls) -> str:
        controls = [
            json.loads(line)
            for line in CONTROLS_AR.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(controls) == 36, "expected 36 SAMA CSF controls"
        fields = ("title", "domain", "text", "principle", "objective")
        return " ".join(
            str(control.get(f, "")) for control in controls for f in fields
        )

    def test_no_flipped_ligatures_remain(self, corpus_text):
        assert arabic_text.repair_lam_alef(corpus_text) == corpus_text

    @pytest.mark.parametrize(
        "bad_form, correct_form",
        [("خالل", "خلال"), ("إبالغ", "إبلاغ"), ("عمالء", "عملاء"),
         ("لألمن", "للأمن"), ("مالحظة", "ملاحظة")],
    )
    def test_known_corruptions_are_absent(self, corpus_text, bad_form, correct_form):
        assert bad_form not in corpus_text, f"{bad_form!r} should read {correct_form!r}"

    @pytest.mark.parametrize("word", ["الإلكترونية", "الآلي", "إلى", "المألوفة", "الثالثة"])
    def test_legitimate_words_survived_the_repair(self, corpus_text, word):
        assert word in corpus_text
