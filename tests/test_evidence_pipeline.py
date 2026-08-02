"""Evidence classification, extraction and chunking."""
import pytest

import compliance_engine
import evidence_pipeline as ep


class TestClassifyEvidence:
    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("mfa.png", "screenshot"),
            ("shot.JPEG", "screenshot"),
            ("matrix.xlsx", "compliance_matrix"),
            ("firewall.yaml", "configuration"),
            ("access_policy.pdf", "policy"),
            ("onboarding_procedure.docx", "procedure"),
            ("internal_audit_2025.pdf", "audit_report"),
            ("unknown.pdf", "policy"),
        ],
    )
    def test_classifies_by_extension_then_filename(self, filename, expected):
        assert ep.classify_evidence(filename) == expected

    def test_user_declared_type_wins(self):
        assert ep.classify_evidence("mfa.png", declared_type="audit_report") == "audit_report"

    def test_unknown_declared_type_is_ignored(self):
        assert ep.classify_evidence("mfa.png", declared_type="not_a_type") == "screenshot"


class TestSpreadsheetChunking:
    """One chunk per row, each carrying its sheet name.

    Regression test: as a single blob a compliance matrix mixes unrelated
    topics, which leaves it moderately similar to *every* control instead of
    sharply similar to the one row that matters.
    """

    SHEET = (
        "[Sheet: Matrix]\n"
        "Area | Implemented | Owner\n"
        "Security awareness training | Yes | HR\n"
        "Vendor risk assessments | Yes | Procurement\n"
    )

    def test_splits_per_row(self):
        chunks = ep._chunk_spreadsheet_rows(self.SHEET)
        assert len(chunks) == 3  # header row + 2 data rows

    def test_each_chunk_carries_the_sheet_name(self):
        chunks = ep._chunk_spreadsheet_rows(self.SHEET)
        assert all(c.startswith("Sheet: Matrix:") for c in chunks)

    def test_topics_stay_isolated(self):
        chunks = ep._chunk_spreadsheet_rows(self.SHEET)
        training = [c for c in chunks if "awareness" in c]
        assert len(training) == 1
        assert "Vendor risk" not in training[0]

    def test_blank_lines_are_dropped(self):
        assert ep._chunk_spreadsheet_rows("[Sheet: S]\n\n  \nRow one\n") == ["Sheet: S: Row one"]


class TestTextExtraction:
    def test_rejects_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            ep.extract_text(b"data", "archive.zip")

    def test_rejects_legacy_doc_with_actionable_message(self):
        with pytest.raises(ValueError, match="re-save"):
            ep.extract_text(b"data", "policy.doc")

    def test_reads_plain_text_files(self):
        assert "min_length: 14" in ep.extract_text(b"min_length: 14\n", "cfg.yaml")

    def test_decodes_utf8_arabic(self):
        text = ep.extract_text("سياسة الأمن".encode("utf-8"), "policy.txt")
        assert "سياسة" in text


class TestBuildEvidenceChunks:
    def test_returns_chunks_manifest_and_language(self):
        files = [{"filename": "cfg.yaml", "content": b"mfa: enabled\nvpn: required\n"}]
        chunks, manifest, language = ep.build_evidence_chunks(
            files, compliance_engine.chunk_text
        )
        assert chunks and manifest
        assert language == "en"

    def test_tags_every_chunk_with_provenance(self):
        files = [{
            "filename": "cfg.yaml",
            "content": b"mfa: enabled\n",
            "evidence_type": "configuration",
            "category": "Access Control",
            "description": "MFA is on",
            "related_control": "3.3.5",
        }]
        chunks, _, _ = ep.build_evidence_chunks(files, compliance_engine.chunk_text)
        chunk = chunks[0]
        assert chunk["source"] == "cfg.yaml"
        assert chunk["evidence_type"] == "configuration"
        assert chunk["category"] == "Access Control"
        assert chunk["related_control"] == "3.3.5"
        # The description is prepended so retrieval can match on it too.
        assert chunk["text"].startswith("MFA is on")

    def test_detects_arabic_package_language(self):
        arabic = "يجب على المؤسسة المالية تطبيق المصادقة متعددة العوامل".encode("utf-8")
        _, _, language = ep.build_evidence_chunks(
            [{"filename": "policy.txt", "content": arabic}], compliance_engine.chunk_text
        )
        assert language == "ar"

    def test_language_is_decided_across_the_whole_package(self):
        """One small English config must not flip an Arabic package."""
        files = [
            {"filename": "policy.txt",
             "content": ("يجب على المؤسسة المالية تطبيق ضوابط الأمن السيبراني وحماية "
                         "الأصول المعلوماتية ومراجعتها دورياً. " * 5).encode("utf-8")},
            {"filename": "cfg.yaml", "content": b"mfa: enabled\n"},
        ]
        _, _, language = ep.build_evidence_chunks(files, compliance_engine.chunk_text)
        assert language == "ar"

    def test_manifest_counts_chunks_per_file(self):
        files = [{"filename": "cfg.yaml", "content": b"a: 1\n"}]
        _, manifest, _ = ep.build_evidence_chunks(files, compliance_engine.chunk_text)
        assert manifest[0]["chunk_count"] >= 1
        assert manifest[0]["filename"] == "cfg.yaml"

    def test_empty_file_is_rejected_by_name(self):
        with pytest.raises(ValueError, match="empty.txt"):
            ep.build_evidence_chunks(
                [{"filename": "empty.txt", "content": b"   "}], compliance_engine.chunk_text
            )


class TestChunkText:
    def test_returns_empty_for_blank_input(self):
        assert compliance_engine.chunk_text("   \n  \n") == []

    def test_keeps_all_content(self):
        text = "\n".join(f"Line {i} of the policy document." for i in range(40))
        joined = " ".join(compliance_engine.chunk_text(text))
        assert "Line 0" in joined and "Line 39" in joined

    def test_splits_long_text_into_several_chunks(self):
        text = "\n".join(f"Sentence number {i} with some padding text." for i in range(60))
        assert len(compliance_engine.chunk_text(text)) > 1
