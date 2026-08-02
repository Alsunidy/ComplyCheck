"""Hybrid retrieval: Chroma vector search + BM25 fused with RRF."""
import numpy as np
import pytest

from retrieval import RRF_K, HybridRetriever, _tokenize

DOCS = [
    "The Cyber Security Committee is headed by an independent senior manager.",
    "MFA is mandatory for all VPN connections and administrative accounts.",
    "The 2023-2025 Security Roadmap details the explicit execution timelines.",
    "Background checks are performed for all new hires prior to onboarding.",
]


@pytest.fixture
def retriever(fake_embeddings):
    return HybridRetriever(DOCS)


class TestTokenizer:
    def test_splits_latin_words_and_lowercases(self):
        assert _tokenize("MFA for VPN") == ["mfa", "for", "vpn"]

    def test_handles_arabic(self):
        assert _tokenize("إدارة الهوية") == ["إدارة", "الهوية"]


class TestSearch:
    def test_returns_requested_number_of_results(self, retriever):
        top, sims = retriever.search("security committee", top_k=2)
        assert len(top) == 2
        assert all(0 <= i < len(DOCS) for i in top)

    def test_similarities_cover_every_document(self, retriever):
        """Callers apply thresholds over all docs, not just the top_k."""
        _, sims = retriever.search("security committee", top_k=2)
        assert len(sims) == len(DOCS)

    def test_exact_keyword_match_ranks_first(self, retriever):
        top, _ = retriever.search("execution timelines", top_k=2)
        assert top[0] == 2

    def test_similarity_scores_are_bounded(self, retriever):
        _, sims = retriever.search("background checks", top_k=1)
        assert np.all(sims >= -1.0001) and np.all(sims <= 1.0001)


class TestRRFFusionIgnoresZeroBM25:
    """A query sharing no term with any document must not be ranked by BM25.

    Regression test for a real defect: when every BM25 score is zero,
    `argsort` still produces an arbitrary order. Feeding that order into the
    fusion with full weight let it outvote a correct semantic hit -- the
    query "multi-factor authentication" returned the hiring-checks sentence
    instead of the MFA one.
    """

    def test_all_zero_bm25_does_not_contribute(self, retriever, monkeypatch):
        monkeypatch.setattr(retriever._bm25, "get_scores", lambda q: np.zeros(len(DOCS)))
        # With BM25 silent, the ranking must be exactly the vector ranking.
        top, sims = retriever.search("security committee", top_k=len(DOCS))
        assert top == [int(i) for i in np.argsort(sims)[::-1]]

    def test_partial_bm25_matches_still_contribute(self, retriever):
        top, _ = retriever.search("onboarding hires", top_k=1)
        assert top[0] == 3


class TestRRFConstant:
    def test_rrf_k_is_the_standard_dampening_value(self):
        assert RRF_K == 60
