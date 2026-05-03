"""
Strategy deduplication utilities.

Prevents near-duplicate strategies from polluting the memory store.
Two strategies are considered duplicates if their cosine similarity
exceeds a configurable threshold (default 0.92).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .strategy_memory import StrategyMemory
from .retriever import Retriever
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Similarity above which two strategies are considered duplicates
DEFAULT_DEDUP_THRESHOLD = 0.92


class DeduplicatingStrategyMemory:
    """
    Wrapper around StrategyMemory that silently drops near-duplicate
    strategies before they are stored.

    Usage
    -----
    memory   = StrategyMemory("results/strategy_memory.db")
    retriever = Retriever(config, memory)
    dedup_mem = DeduplicatingStrategyMemory(memory, retriever)

    # Use dedup_mem.store(...) instead of memory.store(...)
    stored_id = dedup_mem.store(task_description, failure_analysis, strategy_text)
    """

    def __init__(
        self,
        memory: StrategyMemory,
        retriever: Retriever,
        dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    ) -> None:
        self.memory = memory
        self.retriever = retriever
        self.dedup_threshold = dedup_threshold
        self._skipped = 0  # count of duplicates dropped

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        task_description: str,
        failure_analysis: Dict[str, Any],
        strategy_text: str,
    ) -> Optional[int]:
        """
        Embed strategy_text, check for near-duplicates, and store if unique.

        Returns the new row id if stored, or None if skipped as duplicate.
        """
        embedding = self.retriever.embed(strategy_text)

        if self._is_duplicate(embedding, strategy_text):
            self._skipped += 1
            logger.debug(
                "Skipping duplicate strategy (total skipped: %d): %.80s",
                self._skipped,
                strategy_text,
            )
            return None

        row_id = self.memory.store(
            task_description=task_description,
            failure_analysis=failure_analysis,
            strategy_text=strategy_text,
            embedding=embedding,
        )
        logger.info("Stored unique strategy id=%d", row_id)
        return row_id

    def retrieve(
        self,
        query_text: str,
        top_k: int = 3,
        threshold: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """Embed query_text and retrieve top-k strategies."""
        query_embedding = self.retriever.embed(query_text)
        return self.retriever.retrieve(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
        )

    def skipped_count(self) -> int:
        """Number of duplicate strategies dropped so far."""
        return self._skipped

    def count(self) -> int:
        """Total strategies in the underlying store."""
        return self.memory.count()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_duplicate(self, embedding: np.ndarray, strategy_text: str) -> bool:
        """
        Return True if a near-duplicate already exists in memory.

        Checks:
        1. Exact text match (fast path)
        2. High cosine similarity above dedup_threshold (semantic match)
        """
        # Fast path: exact text match via high-similarity retrieval
        candidates = self.memory.retrieve(
            query_embedding=embedding,
            top_k=1,
            similarity_threshold=self.dedup_threshold,
        )
        if not candidates:
            return False

        # Double-check: confirm it's truly a near-duplicate
        top = candidates[0]
        if top["similarity"] >= self.dedup_threshold:
            logger.debug(
                "Near-duplicate found (sim=%.3f): %.60s",
                top["similarity"],
                top["strategy_text"],
            )
            return True

        return False


# ---------------------------------------------------------------------------
# Standalone helper: deduplicate a list of strategy dicts in-memory
# ---------------------------------------------------------------------------

def deduplicate_strategy_list(
    strategies: List[Dict[str, Any]],
    retriever: Retriever,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Given a list of strategy dicts (each with a 'strategy_text' key),
    return a deduplicated list preserving insertion order.

    Useful for cleaning up a batch of strategies before bulk-inserting.
    """
    if not strategies:
        return []

    kept: List[Dict[str, Any]] = []
    kept_embeddings: List[np.ndarray] = []

    for s in strategies:
        text = s.get("strategy_text", "")
        emb = retriever.embed(text)
        is_dup = False
        for kept_emb in kept_embeddings:
            sim = _cosine_similarity(emb, kept_emb)
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(s)
            kept_embeddings.append(emb)

    logger.info(
        "deduplicate_strategy_list: %d → %d strategies (removed %d duplicates)",
        len(strategies),
        len(kept),
        len(strategies) - len(kept),
    )
    return kept


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0
