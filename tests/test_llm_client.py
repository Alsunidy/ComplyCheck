"""The shared OpenAI client: model fallback chain, retries, fail-fast."""
import pytest

import llm_client


class TestRetryDelayParsing:
    @pytest.mark.parametrize(
        "message, expected",
        [
            ("Rate limit reached. Please try again in 1.5s", 1.5),
            ("Please try again in 200ms", 0.2),
            ("try again in 12s. Upgrade for more", 12.0),
            ("some unrelated error", None),
        ],
    )
    def test_honours_the_server_supplied_wait(self, message, expected):
        assert llm_client._parse_retry_delay(message) == expected


class TestFatalAccountErrors:
    """Quota/key problems must fail fast: no model in the chain can help."""

    @pytest.mark.parametrize(
        "message, fatal",
        [
            ("Error code: 429 - {'code': 'insufficient_quota'}", True),
            ("Error code: 401 - {'code': 'invalid_api_key'}", True),
            ("Error code: 429 - rate_limit_exceeded", False),
            ("Error code: 500 - server_error", False),
        ],
    )
    def test_classifies(self, message, fatal):
        assert llm_client._is_fatal_account_error(message) is fatal


class TestModelChain:
    def test_defaults_to_the_built_in_chain(self):
        assert llm_client._model_chain(None) == llm_client.FALLBACK_CHAIN

    def test_preferred_model_is_tried_first(self):
        chain = llm_client._model_chain("gpt-4o")
        assert chain[0] == "gpt-4o"
        assert set(llm_client.FALLBACK_CHAIN).issubset(chain)

    def test_preferred_model_is_not_duplicated(self):
        chain = llm_client._model_chain(llm_client.FALLBACK_CHAIN[1])
        assert len(chain) == len(set(chain))

    def test_every_default_model_supports_vision(self):
        """Screenshots go through the same chain, so all entries must accept images."""
        assert all(m.startswith("gpt-4") for m in llm_client.FALLBACK_CHAIN)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeClient:
    """Records calls and replays a scripted sequence of results/exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.models_tried = []
        self.chat = self
        self.completions = self

    def create(self, model, messages, **kwargs):
        self.models_tried.append(model)
        outcome = self.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(outcome)


@pytest.fixture
def patched_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def install(script):
        fake = _FakeClient(script)
        import openai

        monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: fake)
        return fake

    return install


class TestGenerateJson:
    def test_returns_content_from_the_first_model(self, patched_openai):
        fake = patched_openai(['{"ok": true}'])
        assert llm_client.generate_json("hi") == '{"ok": true}'
        assert fake.models_tried == [llm_client.FALLBACK_CHAIN[0]]

    def test_can_report_which_model_answered(self, patched_openai):
        patched_openai(['{"ok": true}'])
        text, model = llm_client.generate_json("hi", return_model=True)
        assert text == '{"ok": true}'
        assert model == llm_client.FALLBACK_CHAIN[0]

    def test_moves_to_the_next_model_when_one_is_retired(self, patched_openai, monkeypatch):
        monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
        fake = patched_openai([Exception("model_not_found"), '{"ok": true}'])
        assert llm_client.generate_json("hi") == '{"ok": true}'
        assert fake.models_tried == llm_client.FALLBACK_CHAIN[:2]

    def test_fails_fast_on_exhausted_quota(self, patched_openai):
        fake = patched_openai([Exception("Error: insufficient_quota")])
        with pytest.raises(RuntimeError, match="no available quota"):
            llm_client.generate_json("hi")
        # Walking the rest of the chain would be pointless -- and slow.
        assert len(fake.models_tried) == 1

    def test_quota_error_points_at_billing(self, patched_openai):
        patched_openai([Exception("Error: insufficient_quota")])
        with pytest.raises(RuntimeError, match="billing"):
            llm_client.generate_json("hi")

    def test_raises_when_the_whole_chain_fails(self, patched_openai, monkeypatch):
        monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
        attempts = len(llm_client.FALLBACK_CHAIN) * 3
        patched_openai([Exception("server_error")] * attempts)
        with pytest.raises(RuntimeError, match="fallback chain failed"):
            llm_client.generate_json("hi")

    def test_missing_key_is_reported_clearly(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
            llm_client.generate_json("hi")
