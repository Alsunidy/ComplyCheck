"""PDF report generation, live-progress state, and UI copy completeness."""
import pytest

import progress
import report_generator
from ui_text import (
    EVIDENCE_TYPE_LABELS,
    EVIDENCE_TYPES,
    LANGUAGES,
    STAGE_LABELS,
    STATUS_LABELS,
    TEXT,
    t,
)


class TestPdfGeneration:
    def test_produces_a_valid_english_pdf(self, sample_report):
        data = report_generator.generate_compliance_pdf(sample_report)
        assert data.startswith(b"%PDF-")
        assert len(data) > 1000

    def test_produces_a_valid_arabic_pdf(self, sample_report):
        """Arabic needs an embedded TTF, so the file is much larger."""
        sample_report["language"] = "ar"
        sample_report["results"][0]["justification"] = "تُظهر الأدلة اعتماد مجلس الإدارة."
        data = report_generator.generate_compliance_pdf(sample_report)
        assert data.startswith(b"%PDF-")
        assert len(data) > 20000, "an Arabic PDF must embed a font"

    def test_unknown_language_falls_back_to_english(self, sample_report):
        sample_report["language"] = "fr"
        assert report_generator.generate_compliance_pdf(sample_report).startswith(b"%PDF-")

    def test_report_without_the_optional_fields_still_renders(self, sample_report):
        """Older reports have no evidence_used / confidence_score."""
        sample_report.pop("evidence_used")
        sample_report.pop("language")
        for result in sample_report["results"]:
            result.pop("evidence_source", None)
            result.pop("confidence_score", None)
        assert report_generator.generate_compliance_pdf(sample_report).startswith(b"%PDF-")

    def test_empty_result_set_does_not_crash(self, sample_report):
        sample_report["results"] = []
        assert report_generator.generate_compliance_pdf(sample_report).startswith(b"%PDF-")


class TestArabicPdfShaping:
    def test_shaping_reorders_arabic(self):
        import pdf_arabic

        shaped = pdf_arabic.shape("الأمن السيبراني")
        assert shaped != "الأمن السيبراني"   # bidi-reordered for drawing
        assert len(shaped) > 0

    def test_markup_and_entities_survive_shaping(self):
        """Reshaping "&mdash;" turns it into ";mdash&"; tags must be protected."""
        import pdf_arabic

        shaped = pdf_arabic.shape("<b>الإطار</b> &mdash; تقرير")
        assert "<b>" in shaped and "</b>" in shaped
        assert "&mdash;" in shaped

    def test_wrapped_text_keeps_line_order(self):
        """get_display() reverses a whole paragraph, so if ReportLab wrapped it
        afterwards the LINES came out backwards. shape_wrapped pre-wraps."""
        import pdf_arabic

        font = pdf_arabic.ensure_arabic_font()
        long_text = "تُظهر الأدلة المقدمة اعتماد مجلس الإدارة لسياسة الأمن السيبراني ومراجعتها سنوياً"
        wrapped = pdf_arabic.shape_wrapped(long_text, font, 9, 120)
        assert "<br/>" in wrapped, "long text should be split across lines"

    def test_latin_text_passes_through_unchanged(self):
        import pdf_arabic

        assert pdf_arabic.shape("Control 3.1.1") == "Control 3.1.1"


class TestProgressState:
    def setup_method(self):
        progress.finish()

    def test_starts_in_a_known_state(self):
        progress.start("uploading", detail="3 files", total=3)
        snapshot = progress.snapshot()
        assert snapshot["stage"] == "uploading"
        assert snapshot["total"] == 3
        assert snapshot["current"] == 0
        assert snapshot["done"] is False

    def test_advance_increments_the_counter(self):
        progress.start("judging", total=6)
        progress.advance()
        progress.advance()
        assert progress.snapshot()["current"] == 2

    def test_update_changes_only_what_is_given(self):
        progress.start("retrieving", detail="keep me", total=36)
        progress.update(current=10)
        snapshot = progress.snapshot()
        assert snapshot["detail"] == "keep me"
        assert snapshot["stage"] == "retrieving"
        assert snapshot["current"] == 10

    def test_finish_marks_done(self):
        progress.start("judging", total=6)
        progress.finish()
        snapshot = progress.snapshot()
        assert snapshot["stage"] == "complete"
        assert snapshot["done"] is True

    def test_elapsed_is_reported(self):
        progress.start("indexing")
        assert progress.snapshot()["elapsed"] >= 0


class TestUiTextCompleteness:
    """Every string must exist in both languages, or the UI silently falls
    back to English mid-page."""

    def test_both_languages_are_offered(self):
        assert set(LANGUAGES) == {"en", "ar"}

    def test_no_key_is_missing_from_either_language(self):
        assert set(TEXT["en"]) == set(TEXT["ar"])

    def test_no_string_is_left_untranslated(self):
        """Arabic values must not be verbatim copies of the English ones.

        A handful legitimately match: the product name and format-only
        strings.
        """
        allowed_identical = {"app_title"}
        identical = {
            key for key, value in TEXT["en"].items()
            if TEXT["ar"][key] == value and key not in allowed_identical
        }
        assert not identical, f"untranslated: {sorted(identical)}"

    def test_placeholders_match_between_languages(self):
        """A missing {n} would raise KeyError at render time."""
        import re

        for key, english in TEXT["en"].items():
            assert set(re.findall(r"\{(\w+)\}", english)) == \
                   set(re.findall(r"\{(\w+)\}", TEXT["ar"][key])), key

    def test_stage_labels_cover_both_languages(self):
        assert set(STAGE_LABELS["en"]) == set(STAGE_LABELS["ar"])

    def test_status_labels_cover_every_status_code(self):
        for language in ("en", "ar"):
            assert set(STATUS_LABELS[language]) == {"PASS", "PARTIAL", "FAIL"}

    def test_evidence_type_labels_cover_every_type(self):
        for language in ("en", "ar"):
            assert set(EVIDENCE_TYPE_LABELS[language]) == set(EVIDENCE_TYPES)

    def test_ui_evidence_types_match_the_backend(self):
        """The UI list is duplicated rather than imported; it must not drift."""
        import evidence_pipeline

        assert set(EVIDENCE_TYPES) == set(evidence_pipeline.EVIDENCE_TYPES)

    def test_ui_stages_match_what_the_backend_reports(self):
        """Stage keys come from the backend through GET /progress."""
        backend_stages = {
            "uploading", "classifying", "extracting", "analyzing_images",
            "detecting_language", "indexing", "retrieving", "judging", "complete",
        }
        assert backend_stages <= set(STAGE_LABELS["en"])


class TestTranslationLookup:
    def test_returns_the_requested_language(self):
        assert t("ar", "summary") == TEXT["ar"]["summary"]

    def test_formats_placeholders(self):
        assert "3" in t("en", "files_in_package", n=3)

    def test_unknown_language_falls_back_to_english(self):
        assert t("fr", "summary") == TEXT["en"]["summary"]
