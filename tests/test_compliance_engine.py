"""Control loading, verdict parsing and the end-to-end judging pass."""
import pytest

import compliance_engine as ce


class TestLoadControls:
    def test_loads_36_english_controls(self):
        controls = ce._load_controls("en")
        assert len(controls) == 36

    def test_loads_36_arabic_controls(self):
        assert len(ce._load_controls("ar")) == 36

    def test_control_ids_align_across_languages(self):
        english = [c["control_id"] for c in ce._load_controls("en")]
        arabic = [c["control_id"] for c in ce._load_controls("ar")]
        assert english == arabic

    def test_every_control_has_the_fields_retrieval_needs(self):
        for control in ce._load_controls("en"):
            for field in ("control_id", "title", "domain", "text", "principle", "objective"):
                assert control.get(field), f"{control.get('control_id')} missing {field}"

    def test_results_are_cached(self):
        assert ce._load_controls("en") is ce._load_controls("en")


class TestParseVerdict:
    def test_accepts_a_valid_verdict(self):
        verdict = ce._parse_verdict({
            "status_code": "pass",
            "justification": "  Board approval documented.  ",
            "recommendation": "No action required.",
            "confidence_score": 88,
        })
        assert verdict["status_code"] == "PASS"       # normalised to upper case
        assert verdict["justification"] == "Board approval documented."
        assert verdict["confidence_score"] == 88

    @pytest.mark.parametrize("status", ["MAYBE", "", "compliant", None])
    def test_rejects_an_invalid_status(self, status):
        with pytest.raises(ValueError, match="invalid status_code"):
            ce._parse_verdict({"status_code": status})

    @pytest.mark.parametrize(
        "given, expected",
        [(150, 100), (-20, 0), ("not a number", 0), (None, 0), ("75", 75)],
    )
    def test_confidence_is_clamped_to_0_100(self, given, expected):
        verdict = ce._parse_verdict({"status_code": "FAIL", "confidence_score": given})
        assert verdict["confidence_score"] == expected

    def test_missing_prose_fields_become_empty_strings(self):
        verdict = ce._parse_verdict({"status_code": "FAIL"})
        assert verdict["justification"] == ""
        assert verdict["recommendation"] == ""


class TestThresholds:
    def test_source_fairness_bar_is_higher_than_no_match(self):
        """Short evidence sits at a high baseline similarity to everything, so
        the "give every file a chance" pass needs a stricter bar than the
        no-match cutoff."""
        assert ce.SOURCE_FAIRNESS_THRESHOLD > ce.NO_MATCH_THRESHOLD


def _evidence(text, source="policy.pdf", related_control=""):
    return {
        "text": text, "source": source, "evidence_type": "policy",
        "category": "", "description": "", "related_control": related_control,
    }


class TestRunEvidenceComplianceCheck:
    """The full pass, with the retriever and the LLM judge stubbed out."""

    @pytest.fixture
    def stubbed(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class StubRetriever:
            """Ranks everything as a strong match, in input order."""
            def __init__(self, texts):
                self.texts = texts

            def search(self, query, top_k):
                import numpy as np
                sims = np.full(len(self.texts), 0.8)
                return list(range(min(top_k, len(self.texts)))), sims

        monkeypatch.setattr(ce, "HybridRetriever", StubRetriever)

        calls = []

        def fake_judge(batch, language="en"):
            calls.append((len(batch), language))
            return {
                item["control"]["control_id"]: {
                    "status_code": "PASS",
                    "justification": "ok",
                    "recommendation": "No action required.",
                    "confidence_score": 90,
                }
                for item in batch
            }

        monkeypatch.setattr(ce, "_judge_batch", fake_judge)
        return calls

    def test_returns_one_result_per_control(self, stubbed):
        results = ce.run_evidence_compliance_check([_evidence("MFA is enforced.")])
        assert len(results) == 36

    def test_results_carry_the_agreed_schema(self, stubbed):
        results = ce.run_evidence_compliance_check([_evidence("MFA is enforced.")])
        required = {
            "control_id", "control_domain", "control_text", "status_code",
            "status_label", "matched_policy_excerpt", "justification",
            "recommendation", "evidence_source", "confidence_score",
        }
        assert required.issubset(results[0])

    def test_status_label_matches_status_code(self, stubbed):
        results = ce.run_evidence_compliance_check([_evidence("MFA is enforced.")])
        for r in results:
            assert r["status_label"] == ce.STATUS_LABELS[r["status_code"]]

    def test_controls_are_judged_in_batches(self, stubbed):
        ce.run_evidence_compliance_check([_evidence("MFA is enforced.")])
        assert len(stubbed) == 6  # 36 controls / batch size 6
        assert all(size <= ce.JUDGE_BATCH_SIZE for size, _ in stubbed)

    def test_language_reaches_the_judge(self, stubbed):
        ce.run_evidence_compliance_check([_evidence("تطبيق المصادقة")], language="ar")
        assert all(language == "ar" for _, language in stubbed)

    def test_evidence_source_is_recorded(self, stubbed):
        results = ce.run_evidence_compliance_check(
            [_evidence("MFA is enforced.", source="mfa.png")]
        )
        assert results[0]["evidence_source"] == ["mfa.png"]

    def test_missing_api_key_is_reported_before_any_work(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
            ce.run_evidence_compliance_check([_evidence("anything")])


class TestNoRelevantEvidence:
    """With nothing relevant retrieved, the engine must not call the LLM at
    all -- it reports an honest FAIL instead of inviting a fabricated one."""

    @pytest.fixture
    def stubbed_no_match(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        class ColdRetriever:
            def __init__(self, texts):
                self.texts = texts

            def search(self, query, top_k):
                import numpy as np
                return [0], np.full(len(self.texts), 0.01)  # far below threshold

        monkeypatch.setattr(ce, "HybridRetriever", ColdRetriever)

        judged = []
        monkeypatch.setattr(
            ce, "_judge_batch",
            lambda batch, language="en": judged.append(batch) or {},
        )
        return judged

    def test_llm_is_never_called(self, stubbed_no_match):
        ce.run_evidence_compliance_check([_evidence("Unrelated text about catering.")])
        assert stubbed_no_match == []

    def test_every_control_fails_with_an_honest_message(self, stubbed_no_match):
        results = ce.run_evidence_compliance_check([_evidence("Unrelated text.")])
        assert all(r["status_code"] == "FAIL" for r in results)
        assert all(not r["evidence_source"] for r in results)
        assert "No relevant content" in results[0]["justification"]

    def test_arabic_no_match_message_is_arabic(self, stubbed_no_match):
        results = ce.run_evidence_compliance_check(
            [_evidence("نص غير ذي صلة")], language="ar"
        )
        assert "لم يُعثر" in results[0]["justification"]
