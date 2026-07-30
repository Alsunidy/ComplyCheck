"""
engine/retrieval.py

Hybrid retrieval over the evidence chunks: semantic vector search (ChromaDB)
fused with lexical keyword search (BM25) via Reciprocal Rank Fusion.

Why hybrid? The two retrievers fail differently, and compliance evidence
needs both:
  - Vector search (Chroma + multilingual fastembed embeddings) matches by
    meaning, so "staff security education" still finds "awareness training"
    -- but it can rank an exact-term match surprisingly low.
  - BM25 matches exact terms, so control IDs, acronyms (MFA, SAMA, KRI) and
    precise phrases like "execution timelines" are found even when the
    embedding model underweights them -- but it knows nothing about synonyms.
Reciprocal Rank Fusion combines the two rankings without needing their
scores to be comparable: each document earns 1/(k + rank) from every list
that ranks it, so items ranked well by BOTH retrievers rise to the top.

The Chroma collection lives in memory for the duration of one compliance
run (evidence is per-session data); the SAMA controls corpus itself is small
and static, so nothing needs persisting between runs.
"""
import re
import uuid

import numpy as np

import progress
from common.embeddings import embed_texts, embed_query

# \w+ with UNICODE matches Arabic and Latin word characters alike.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Standard RRF dampening constant (Cormack et al.); larger k flattens the
# difference between adjacent ranks.
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    """Indexes a list of texts once, then serves fused vector+BM25 queries."""

    def __init__(self, texts: list[str]):
        import chromadb

        self._texts = texts

        # Vector index: an in-memory Chroma collection using cosine space.
        # Embeddings are computed with the shared fastembed model so the
        # static controls and per-run evidence always share one vector space.
        self._client = chromadb.Client()
        self._collection = self._client.create_collection(
            name=f"evidence_{uuid.uuid4().hex[:8]}",
            metadata={"hnsw:space": "cosine"},
        )
        progress.update(stage="indexing", detail=f"{len(texts)}")
        self._collection.add(
            ids=[str(i) for i in range(len(texts))],
            embeddings=embed_texts(texts),
            documents=texts,
        )

        # Lexical index over the same texts.
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([_tokenize(t) for t in texts])

    def search(self, query: str, top_k: int) -> tuple[list[int], np.ndarray]:
        """Return (fused top_k chunk indices, cosine similarity per chunk).

        The similarity array covers ALL chunks (not just the top_k) so the
        caller can apply threshold logic (no-match detection, per-source
        fairness) on the raw semantic scores.
        """
        n = len(self._texts)

        result = self._collection.query(
            query_embeddings=[embed_query(query)],
            n_results=n,
            include=["distances"],
        )
        vector_sims = np.zeros(n)
        for chunk_id, distance in zip(result["ids"][0], result["distances"][0]):
            vector_sims[int(chunk_id)] = 1.0 - distance  # cosine distance -> similarity

        bm25_scores = np.asarray(self._bm25.get_scores(_tokenize(query)))

        rrf = np.zeros(n)
        for rank, idx in enumerate(np.argsort(vector_sims)[::-1]):
            rrf[idx] += 1.0 / (RRF_K + rank + 1)

        # Only documents BM25 actually matched may contribute. A zero BM25
        # score means "no shared term", and a control's wording often shares
        # no term with the evidence at all (query "multi-factor
        # authentication" vs evidence "MFA"). Ranking those zeros anyway
        # would feed an arbitrary tie order into the fusion with full weight
        # and can outvote a correct semantic hit.
        bm25_ranked = [i for i in np.argsort(bm25_scores)[::-1] if bm25_scores[i] > 0]
        for rank, idx in enumerate(bm25_ranked):
            rrf[idx] += 1.0 / (RRF_K + rank + 1)

        fused_top = [int(i) for i in np.argsort(rrf)[::-1][:top_k]]
        return fused_top, vector_sims
