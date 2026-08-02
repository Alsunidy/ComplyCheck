"""End-to-end checks that call the real OpenAI API.

Skipped unless OPENAI_API_KEY is set, so the default `pytest` run stays fast,
offline and free. Run them explicitly with:

    pytest -m integration

They cost a few cents and take about a minute (the first run also downloads
the ~220MB embedding model).
"""
import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY is not set",
    ),
]


@pytest.fixture(scope="module")
def english_evidence(sample_evidence_dir):
    path = sample_evidence_dir / "sample_policy.pdf"
    if not path.exists():
        pytest.skip("sample_policy.pdf is missing")
    return [{"filename": path.name, "content": path.read_bytes()}]


def test_full_pipeline_produces_a_report(english_evidence):
    """Upload -> chunk -> retrieve -> judge, against the real model."""
    import compliance_engine as ce
    import evidence_pipeline as ep

    chunks, manifest, language = ep.build_evidence_chunks(english_evidence, ce.chunk_text)
    assert chunks and manifest
    assert language == "en"

    results = ce.run_evidence_compliance_check(chunks, language=language)

    assert len(results) == 36
    for result in results:
        assert result["status_code"] in {"PASS", "PARTIAL", "FAIL"}
        assert result["justification"], f"{result['control_id']} has no justification"
        assert 0 <= result["confidence_score"] <= 100

    # A real policy should not come back uniformly graded -- that would mean
    # the judge is ignoring the evidence.
    assert len({r["status_code"] for r in results}) > 1


def test_chat_answers_from_the_framework():
    import chat_handler as ch

    result = ch.answer_report_question("What does SAMA require for incident management?")
    assert result["cited_control_ids"], "an answer should cite the control it used"
    assert len(result["answer"]) > 40


def test_chat_refuses_questions_outside_the_corpus():
    """Grounding check: the model knows this, but it is not in the context."""
    import chat_handler as ch

    result = ch.answer_report_question("What is the capital of France?")
    assert "don't have that information" in result["answer"].lower()
    assert result["cited_control_ids"] == []


def test_arabic_question_is_answered_in_arabic():
    import arabic_text
    import chat_handler as ch

    result = ch.answer_report_question("ما متطلبات ساما لإدارة الحوادث السيبرانية؟")
    assert arabic_text.detect_language(result["answer"]) == "ar"
    assert "3.3.15" in result["cited_control_ids"]
