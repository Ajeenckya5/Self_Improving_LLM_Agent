"""
Semantic retrieval using sentence-transformers embeddings.
Wraps StrategyMemory to provide embed + retrieve in one place.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import numpy as np

from .strategy_memory import StrategyMemory
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Loads a sentence-transformers model once and provides:
      - embed(text) -> np.ndarray
      - retrieve(query_embedding, top_k, threshold) -> List[dict]
    """

    def __init__(self, config: Dict[str, Any], memory: StrategyMemory):
        self.config = config
        self.memory = memory
        self._model_name: str = config.get("model", {}).get("embedding", "all-MiniLM-L6-v2")
        self._model = None  # lazy load

    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        model = self._get_model()
        embedding = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return embedding[0]

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        threshold: float = 0.65,
        high_similarity_threshold: float = 0.92,
    ) -> List[Dict[str, Any]]:
        """
        Return best-1 strategy (if above threshold) plus any others
        with similarity >= high_similarity_threshold, capped at top_k.
        """
        candidates = self.memory.retrieve(
            query_embedding=query_embedding,
            top_k=50,
            similarity_threshold=threshold,
        )
        if not candidates:
            return []
        selected = [candidates[0]]
        for c in candidates[1:]:
            if c["similarity"] >= high_similarity_threshold:
                selected.append(c)
        return selected[:top_k]

    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model: %s", self._model_name)
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed; falling back to random embeddings. "
                    "Install with: pip install sentence-transformers"
                )
                self._model = _FakeEmbeddingModel(dim=384)
        return self._model


class _FakeEmbeddingModel:
    """Deterministic fake embedder for CI / testing without GPU."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        rng = np.random.default_rng(seed=abs(hash(texts[0])) % (2**32))
        vectors = rng.standard_normal((len(texts), self.dim)).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
