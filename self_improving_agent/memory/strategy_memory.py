"""
SQLite-backed persistent strategy store.
"""

from __future__ import annotations

import io
import json
import pickle
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_description TEXT,
    failure_type TEXT,
    strategy_text TEXT,
    tags TEXT,
    embedding BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    use_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0
);
"""


class StrategyMemory:
    """
    Thread-safe SQLite store for corrective strategies.

    Usage
    -----
    memory = StrategyMemory("results/strategy_memory.db")
    memory.store(task_description, failure_analysis, strategy_text, embedding)
    strategies = memory.retrieve(query_embedding, top_k=3, similarity_threshold=0.65)
    memory.update_outcome(strategy_id, success=True)
    """

    def __init__(self, db_path: str = "results/strategy_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        task_description: str,
        failure_analysis: Dict[str, Any],
        strategy_text: str,
        embedding: np.ndarray,
    ) -> int:
        """
        Persist a strategy and return its row id.
        """
        tags = failure_analysis.get("tags", [])
        if isinstance(tags, list):
            tags_json = json.dumps(tags)
        else:
            tags_json = json.dumps([str(tags)])

        failure_type = failure_analysis.get("failure_type", "other")
        embedding_blob = self._serialize_embedding(embedding)

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO strategies
                    (task_description, failure_type, strategy_text, tags, embedding)
                VALUES (?, ?, ?, ?, ?)
                """,
                (task_description, failure_type, strategy_text, tags_json, embedding_blob),
            )
            row_id = cursor.lastrowid

        logger.info("Stored strategy id=%d (failure_type=%s)", row_id, failure_type)
        return row_id

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        similarity_threshold: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-k most similar strategies above similarity_threshold.
        """
        rows = self._fetch_all_with_embeddings()
        if not rows:
            return []

        results = []
        for row in rows:
            stored_emb = self._deserialize_embedding(row["embedding"])
            if stored_emb is None:
                continue
            sim = self._cosine_similarity(query_embedding, stored_emb)
            if sim >= similarity_threshold:
                results.append({
                    "id": row["id"],
                    "task_description": row["task_description"],
                    "failure_type": row["failure_type"],
                    "strategy_text": row["strategy_text"],
                    "tags": json.loads(row["tags"] or "[]"),
                    "similarity": float(sim),
                    "use_count": row["use_count"],
                    "success_count": row["success_count"],
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        top = results[:top_k]

        # Increment use_count for returned strategies
        if top:
            ids = [r["id"] for r in top]
            with self._connect() as conn:
                conn.executemany(
                    "UPDATE strategies SET use_count = use_count + 1 WHERE id = ?",
                    [(rid,) for rid in ids],
                )

        return top

    def update_outcome(self, strategy_id: int, success: bool) -> None:
        """Track whether a retrieved strategy led to task success."""
        with self._connect() as conn:
            if success:
                conn.execute(
                    "UPDATE strategies SET success_count = success_count + 1 WHERE id = ?",
                    (strategy_id,),
                )

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM strategies").fetchone()
        return row[0] if row else 0

    def all_strategies(self) -> List[Dict[str, Any]]:
        """Return all strategies without embeddings (for inspection)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, task_description, failure_type, strategy_text, tags, "
                "created_at, use_count, success_count FROM strategies ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def clear(self) -> None:
        """Delete all stored strategies (useful for ablation studies)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM strategies")
        logger.info("Cleared all strategies from memory.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _fetch_all_with_embeddings(self) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT id, task_description, failure_type, strategy_text, tags, "
                "embedding, use_count, success_count FROM strategies"
            ).fetchall()

    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, embedding)
        return buf.getvalue()

    @staticmethod
    def _deserialize_embedding(blob: bytes) -> Optional[np.ndarray]:
        if blob is None:
            return None
        try:
            buf = io.BytesIO(blob)
            return np.load(buf)
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a.flatten().astype(np.float32)
        b = b.flatten().astype(np.float32)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
