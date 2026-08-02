"""FastAPI endpoint contracts, exercised with TestClient (no real LLM)."""
import json

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def stub_engine(monkeypatch, sample_report):
    """Replace the LLM-backed engine so endpoint behaviour can be tested."""
    monkeypatch.setattr(
        main.compliance_engine, "run_evidence_compliance_check",
        lambda chunks, language="en": sample_report["results"],
    )
    return sample_report


class TestRoot:
    def test_reports_service_status(self, client):
        body = client.get("/").json()
        assert body["status"] == "running"


class TestUploadEvidence:
    def test_accepts_a_multi_file_package(self, client):
        response = client.post("/upload-evidence", files=[
            ("files", ("policy.txt", b"MFA is enforced for all admin accounts.", "text/plain")),
            ("files", ("cfg.yaml", b"mfa: enabled\n", "text/yaml")),
        ])
        assert response.status_code == 200
        body = response.json()
        assert len(body["files"]) == 2
        assert body["total_chunks"] >= 2

    def test_reports_the_detected_language(self, client):
        response = client.post("/upload-evidence", files=[
            ("files", ("policy.txt",
                       "يجب على المؤسسة المالية تطبيق ضوابط الأمن السيبراني".encode("utf-8"),
                       "text/plain")),
        ])
        assert response.json()["language"] == "ar"

    def test_per_file_metadata_is_applied(self, client):
        metadata = json.dumps([{
            "filename": "cfg.yaml", "type": "configuration",
            "category": "Access Control", "description": "MFA on",
            "related_control": "3.3.5",
        }])
        response = client.post(
            "/upload-evidence",
            files=[("files", ("cfg.yaml", b"mfa: enabled\n", "text/yaml"))],
            data={"metadata": metadata},
        )
        entry = response.json()["files"][0]
        assert entry["type"] == "configuration"
        assert entry["related_control"] == "3.3.5"

    def test_rejects_an_unsupported_file_type(self, client):
        response = client.post("/upload-evidence", files=[
            ("files", ("archive.zip", b"PK\x03\x04", "application/zip")),
        ])
        assert response.status_code == 400
        assert ".zip" in response.json()["detail"]

    def test_rejects_an_empty_file(self, client):
        response = client.post("/upload-evidence", files=[
            ("files", ("empty.txt", b"", "text/plain")),
        ])
        assert response.status_code == 400

    def test_rejects_malformed_metadata(self, client):
        response = client.post(
            "/upload-evidence",
            files=[("files", ("cfg.yaml", b"a: 1\n", "text/yaml"))],
            data={"metadata": "not json"},
        )
        assert response.status_code == 400


class TestUploadPolicyBackwardCompatibility:
    """The original single-file endpoint must keep working."""

    def test_accepts_a_single_document(self, client):
        response = client.post("/upload-policy", files={
            "file": ("policy.pdf", _tiny_pdf(), "application/pdf")
        })
        assert response.status_code == 200
        assert response.json()["filename"] == "policy.pdf"

    def test_still_rejects_non_policy_types(self, client):
        response = client.post("/upload-policy", files={
            "file": ("shot.png", b"\x89PNG\r\n", "image/png")
        })
        assert response.status_code == 400


class TestRunComplianceCheck:
    def test_requires_evidence_first(self, client, monkeypatch, stub_engine):
        monkeypatch.setattr(main, "EVIDENCE_PACKAGE", {})
        response = client.post("/run-compliance-check", data={"standard": "SAMA"})
        assert response.status_code == 400
        assert "No evidence uploaded" in response.json()["detail"]

    def test_rejects_an_unknown_standard(self, client):
        response = client.post("/run-compliance-check", data={"standard": "ISO"})
        assert response.status_code == 400
        assert "ISO" in response.json()["detail"]

    def test_returns_a_report_after_upload(self, client, stub_engine):
        client.post("/upload-evidence", files=[
            ("files", ("policy.txt", b"MFA is enforced.", "text/plain")),
        ])
        response = client.post("/run-compliance-check", data={"standard": "SAMA"})
        assert response.status_code == 200
        body = response.json()
        assert body["standard"] == "SAMA"
        assert body["results"]
        assert body["evidence_used"]

    def test_report_is_retrievable_by_id(self, client, stub_engine):
        client.post("/upload-evidence", files=[
            ("files", ("policy.txt", b"MFA is enforced.", "text/plain")),
        ])
        report_id = client.post("/run-compliance-check",
                                data={"standard": "SAMA"}).json()["report_id"]
        assert client.get(f"/report/{report_id}").status_code == 200


class TestGetReport:
    def test_unknown_id_is_404(self, client):
        assert client.get("/report/does-not-exist").status_code == 404


class TestProgress:
    def test_exposes_the_current_stage(self, client):
        body = client.get("/progress").json()
        assert {"stage", "current", "total", "done", "elapsed"} <= set(body)


class TestChat:
    @pytest.fixture(autouse=True)
    def stub_answer(self, monkeypatch):
        monkeypatch.setattr(
            main.chat_handler, "answer_report_question",
            lambda **kwargs: {"answer": "stub", "cited_control_ids": ["3.1.1"]},
        )

    def test_works_without_a_report(self, client):
        response = client.post("/chat", json={"question": "What is required?"})
        assert response.status_code == 200
        assert response.json()["answer"] == "stub"

    def test_unknown_report_id_is_404(self, client):
        response = client.post("/chat", json={"question": "q", "report_id": "nope"})
        assert response.status_code == 404

    def test_question_is_required(self, client):
        assert client.post("/chat", json={}).status_code == 422

    def test_history_is_accepted_and_forwarded(self, client, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            main.chat_handler, "answer_report_question",
            lambda **kwargs: seen.update(kwargs) or {"answer": "ok", "cited_control_ids": []},
        )
        client.post("/chat", json={
            "question": "elaborate",
            "history": [{"role": "user", "content": "earlier question"}],
        })
        assert seen["history"] == [{"role": "user", "content": "earlier question"}]

    def test_invalid_history_role_is_rejected(self, client):
        response = client.post("/chat", json={
            "question": "q", "history": [{"role": "system", "content": "x"}],
        })
        assert response.status_code == 422


def _tiny_pdf() -> bytes:
    """Smallest PDF pymupdf will parse, generated on the fly."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "MFA is enforced for all administrative accounts.")
    data = doc.tobytes()
    doc.close()
    return data
