"""FAISS cosine retrieval with lexical or optional neural embeddings."""
import time
import math
from pipeline.configuration import positive_integer
from collections import defaultdict

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class FaissVectorStore:
    def __init__(self, max_features: int = 8192, embedder=None):
        positive_integer(max_features, "max_features")
        self.max_features = max_features
        self.embedder = embedder
        self.vectorizer = None
        self.index = None
        self.chunks_metadata = []
        self.is_fitted = False

    @staticmethod
    def _normalize(values, rows):
        try:
            matrix = np.asarray(values, dtype="float32")
        except (ValueError, TypeError) as exc:
            raise ValueError("Embedding provider returned non-numeric values.") from exc
        if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] == 0:
            raise ValueError("Embedding provider returned an invalid matrix shape.")
        if not np.isfinite(matrix).all():
            raise ValueError("Embedding provider returned non-finite values.")
        matrix = np.ascontiguousarray(matrix)
        faiss.normalize_L2(matrix)
        return matrix

    def build_index(self, all_chunks: list[dict]) -> dict:
        started = time.perf_counter()
        if not all_chunks:
            self.index = self.vectorizer = None
            self.chunks_metadata = []
            self.is_fitted = False
            return {"indexed_count": 0, "indexing_latency_ms": 0, "dimensions": 0}
        texts = [chunk["text"] for chunk in all_chunks]
        vectorizer = None
        if self.embedder:
            vectors = self._normalize(self.embedder(texts), len(texts))
            if np.any(np.linalg.norm(vectors, axis=1) == 0):
                raise ValueError("Embedding provider returned a zero document vector. Index was not updated.")
        else:
            vectorizer = TfidfVectorizer(max_features=self.max_features,
                                         stop_words="english", sublinear_tf=True)
            try:
                vectors = self._normalize(vectorizer.fit_transform(texts).toarray(), len(texts))
            except ValueError as exc:
                raise ValueError("No searchable vocabulary in these PDFs.") from exc
        if not np.any(vectors):
            raise ValueError("Embedding provider returned only zero vectors.")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        # Commit only after every step succeeds; failed rebuilds keep a coherent old index.
        self.index, self.vectorizer = index, vectorizer
        self.chunks_metadata = [dict(chunk) for chunk in all_chunks]
        self.is_fitted = True
        return {"indexed_count": len(texts), "dimensions": index.d,
                "indexing_latency_ms": round((time.perf_counter() - started) * 1000, 2)}

    def search(self, query: str, top_k: int = 6, min_score: float = 0.05,
               balance_papers: bool = True, doc_ids: list[str] | None = None) -> tuple[list[dict], float]:
        positive_integer(top_k, "top_k")
        if isinstance(min_score, bool) or not isinstance(min_score, (int, float)) or not math.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("top_k must be positive and min_score must be between 0 and 1.")
        if not query.strip() or not self.is_fitted or self.index is None:
            return [], 0.0
        started = time.perf_counter()
        if self.embedder and hasattr(self.embedder, "embed_query"):
            values = [self.embedder.embed_query(query)]
        else:
            values = (self.embedder([query]) if self.embedder
                      else self.vectorizer.transform([query]).toarray())
        vector = self._normalize(values, 1)
        if vector.shape[1] != self.index.d:
            raise ValueError("Embedding dimensions changed. Rebuild the index.")
        if not np.any(vector):
            return [], round((time.perf_counter() - started) * 1000, 2)
        scores, indices = self.index.search(vector, self.index.ntotal)
        groups = defaultdict(list)
        ranked = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score <= 0 or score < min_score:
                continue
            hit = dict(self.chunks_metadata[idx], similarity_score=round(float(score), 4))
            if doc_ids is not None and hit.get("doc_id", hit["filename"]) not in doc_ids:
                continue
            ranked.append(hit)
            groups[hit.get("doc_id", hit["filename"])].append(hit)
        if balance_papers:
            # Round-robin gives each relevant paper a chance before a second passage.
            ranked = []
            for depth in range(top_k):
                for hits in groups.values():
                    if depth < len(hits):
                        ranked.append(hits[depth])
                if len(ranked) >= top_k:
                    break
        return ranked[:top_k], round((time.perf_counter() - started) * 1000, 2)
