"""Shared test setup.

The app is not an installed package -- backend/ and backend/engine/ are put
on sys.path at runtime by main.py -- so the tests do the same thing here.

Design rule for this suite: the default run must need no API key and no
network. Anything that would call OpenAI is monkeypatched, and the one test
that exercises the real model is marked `integration` and skipped unless
OPENAI_API_KEY is set.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
ENGINE = BACKEND / "engine"
FRONTEND = ROOT / "frontend"

for path in (BACKEND, ENGINE, FRONTEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: hits the real OpenAI API (needs OPENAI_API_KEY)"
    )


@pytest.fixture(scope="session")
def sample_evidence_dir() -> Path:
    return ROOT / "sample_evidence"


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Deterministic stand-in for the multilingual embedding model.

    The real model is a ~220MB download and is far too slow for unit tests.
    Vectors are derived from character content so that texts sharing words
    land near each other -- enough to exercise ranking without pretending to
    be semantically meaningful.
    """
    import retrieval

    def _vector(text: str) -> list[float]:
        vec = [0.0] * 16
        for token in text.lower().split():
            vec[hash(token) % 16] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    monkeypatch.setattr(retrieval, "embed_texts", lambda texts: [_vector(t) for t in texts])
    monkeypatch.setattr(retrieval, "embed_query", lambda text: _vector(text))
    return _vector


@pytest.fixture
def sample_report() -> dict:
    """A minimal report in the shape the API returns."""
    return {
        "report_id": "test-report-0001",
        "standard": "SAMA",
        "generated_at": "2026-07-30T10:00:00+00:00",
        "source_document": "policy.pdf",
        "language": "en",
        "results": [
            {
                "control_id": "3.1.1",
                "control_domain": "Cyber Security Governance",
                "control_text": "A cyber security governance structure shall be defined.",
                "status_code": "PASS",
                "status_label": "Compliant",
                "matched_policy_excerpt": "The board approves the policy annually.",
                "justification": "Board approval is documented.",
                "recommendation": "No action required.",
                "evidence_source": ["policy.pdf"],
                "confidence_score": 90,
            },
            {
                "control_id": "3.3.5",
                "control_domain": "Identity and Access Management",
                "control_text": "MFA shall be enforced for privileged access.",
                "status_code": "FAIL",
                "status_label": "Non-Compliant",
                "matched_policy_excerpt": "No matching content found in the submitted evidence.",
                "justification": "No evidence of MFA enforcement was found.",
                "recommendation": "Enable MFA for all privileged accounts.",
                "evidence_source": [],
                "confidence_score": 85,
            },
        ],
        "evidence_used": [
            {
                "filename": "policy.pdf",
                "type": "policy",
                "category": "",
                "description": "",
                "related_control": "",
                "chunk_count": 4,
            }
        ],
    }
