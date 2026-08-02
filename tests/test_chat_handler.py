"""Chat: grounding, citation filtering, language routing, follow-up rewriting."""
import json

import pytest

import chat_handler as ch


@pytest.fixture
def fake_llm(monkeypatch):
    """Capture prompts and replay canned JSON answers."""
    captured = {}

    def install(payload, *, fail=None):
        def fake_generate_json(contents, system_instruction=None, **kwargs):
            captured["contents"] = contents
            captured["system_instruction"] = system_instruction or ""
            if fail:
                raise fail
            return json.dumps(payload)

        monkeypatch.setattr(ch.llm_client, "generate_json", fake_generate_json)
        return captured

    return install


@pytest.fixture
def stub_retriever(monkeypatch):
    """Avoid the 220MB embedding model: return the first N controls."""
    class Stub:
        def __init__(self, texts):
            self.texts = texts

        def search(self, query, top_k):
            return list(range(min(top_k, len(self.texts)))), None

    monkeypatch.setattr(ch, "_retriever_cache", {})
    import retrieval
    monkeypatch.setattr(retrieval, "HybridRetriever", Stub)
    return Stub


class TestFormatHistory:
    def test_empty_history_is_empty_string(self):
        assert ch._format_history(None) == ""
        assert ch._format_history([]) == ""

    def test_labels_each_turn(self):
        text = ch._format_history([
            {"role": "user", "content": "What is 3.1.1?"},
            {"role": "assistant", "content": "Governance."},
        ])
        assert "User: What is 3.1.1?" in text
        assert "Assistant: Governance." in text

    def test_only_recent_turns_are_kept(self):
        history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
        lines = ch._format_history(history).splitlines()
        assert len(lines) == ch.MAX_HISTORY_MESSAGES
        assert "q19" in lines[-1]

    def test_blank_messages_are_skipped(self):
        assert ch._format_history([{"role": "user", "content": ""}]) == ""


class TestRewriteQuery:
    """A follow-up shares no words with its topic, so it must be rewritten
    before retrieval. Regression: "فصل اكثر" (elaborate more) was searched
    literally and returned the control about SEPARATING units, because
    "فصل" means separation."""

    def test_without_history_the_question_is_used_as_is(self):
        assert ch._rewrite_query("elaborate more", None) == "elaborate more"

    def test_follow_up_is_expanded_using_the_conversation(self, fake_llm):
        fake_llm({"query": "SAMA requirements for incident management"})
        rewritten = ch._rewrite_query("elaborate more", [
            {"role": "user", "content": "What does SAMA require for incident management?"},
            {"role": "assistant", "content": "Control 3.3.15 requires..."},
        ])
        assert rewritten == "SAMA requirements for incident management"

    def test_rewriting_failure_falls_back_to_the_original(self, fake_llm):
        fake_llm({}, fail=RuntimeError("LLM down"))
        history = [{"role": "user", "content": "anything"}]
        assert ch._rewrite_query("elaborate more", history) == "elaborate more"

    def test_empty_rewrite_falls_back_to_the_original(self, fake_llm):
        fake_llm({"query": "   "})
        history = [{"role": "user", "content": "anything"}]
        assert ch._rewrite_query("elaborate more", history) == "elaborate more"


class TestAnswerFromTheFramework:
    def test_answers_and_cites_retrieved_controls(self, fake_llm, stub_retriever):
        fake_llm({"answer": "Control 3.1.1 covers governance.",
                  "cited_control_ids": ["3.1.1"]})
        result = ch.answer_report_question("What is required for governance?")
        assert result["answer"] == "Control 3.1.1 covers governance."
        assert result["cited_control_ids"] == ["3.1.1"]

    def test_invented_citations_are_dropped(self, fake_llm, stub_retriever):
        """The model may name a control that was never in its context."""
        fake_llm({"answer": "See 9.9.9.", "cited_control_ids": ["9.9.9"]})
        assert ch.answer_report_question("anything")["cited_control_ids"] == []

    def test_context_contains_control_text(self, fake_llm, stub_retriever):
        captured = fake_llm({"answer": "ok", "cited_control_ids": []})
        ch.answer_report_question("governance?")
        assert "SAMA CSF control" in captured["contents"]

    def test_malformed_json_yields_a_graceful_message(self, monkeypatch, stub_retriever):
        monkeypatch.setattr(ch.llm_client, "generate_json", lambda **k: "not json")
        result = ch.answer_report_question("anything")
        assert "couldn't produce a reliable answer" in result["answer"]
        assert result["cited_control_ids"] == []

    def test_llm_outage_is_surfaced_not_swallowed(self, monkeypatch, stub_retriever):
        def boom(**kwargs):
            raise RuntimeError("all models failed")
        monkeypatch.setattr(ch.llm_client, "generate_json", boom)
        assert "temporarily unavailable" in ch.answer_report_question("q")["answer"]


class TestLanguageRouting:
    def test_arabic_question_gets_the_arabic_instruction(self, fake_llm, stub_retriever):
        captured = fake_llm({"answer": "إجابة", "cited_control_ids": []})
        ch.answer_report_question("ما متطلبات إدارة الحوادث؟")
        assert "Arabic" in captured["system_instruction"]

    def test_english_question_gets_no_arabic_instruction(self, fake_llm, stub_retriever):
        captured = fake_llm({"answer": "answer", "cited_control_ids": []})
        ch.answer_report_question("What about incident management?")
        assert "formal Modern Standard Arabic" not in captured["system_instruction"]

    def test_report_language_wins_over_the_question(self, fake_llm, sample_report):
        """An Arabic report stays Arabic even if a question is typed in English."""
        captured = fake_llm({"answer": "إجابة", "cited_control_ids": []})
        sample_report["language"] = "ar"
        ch.answer_report_question("what failed?", report=sample_report)
        assert "Arabic" in captured["system_instruction"]


class TestAnswerAboutAReport:
    def test_whole_report_becomes_context(self, fake_llm, sample_report):
        captured = fake_llm({"answer": "3.3.5 needs attention.",
                             "cited_control_ids": ["3.3.5"]})
        result = ch.answer_report_question("Which controls need attention?",
                                           report=sample_report)
        assert result["cited_control_ids"] == ["3.3.5"]
        assert "Non-Compliant" in captured["contents"]

    def test_control_id_narrows_the_context(self, fake_llm, sample_report):
        captured = fake_llm({"answer": "It passed.", "cited_control_ids": ["3.1.1"]})
        ch.answer_report_question("Why?", report=sample_report, control_id="3.1.1")
        assert "3.1.1" in captured["contents"]
        assert "3.3.5" not in captured["contents"]

    def test_unknown_control_id_admits_ignorance(self, sample_report):
        result = ch.answer_report_question("Why?", report=sample_report, control_id="9.9.9")
        assert "don't have that information" in result["answer"]
        assert result["cited_control_ids"] == []

    def test_history_is_included_in_the_prompt(self, fake_llm, sample_report):
        captured = fake_llm({"answer": "ok", "cited_control_ids": []})
        ch.answer_report_question(
            "and the other one?", report=sample_report,
            history=[{"role": "user", "content": "tell me about 3.1.1"}],
        )
        assert "Conversation so far" in captured["contents"]
