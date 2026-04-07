"""Strategy memory store indexed by embeddings."""

import json
from pathlib import Path
from typing import Any

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


class _FallbackStrategyMemory:
    """In-memory fallback when chromadb/sentence-transformers unavailable."""

    def __init__(self, persist_path: Path | str | None = None, embedding_model: str = ""):
        self._strategies: list[dict] = []

    def add(self, task_id: str, task_description: str, failure_category: str,
            corrective_strategy: str, metadata: dict | None = None) -> None:
        self._strategies.append({
            "task_id": task_id,
            "task_description": task_description,
            "failure_category": failure_category,
            "corrective_strategy": corrective_strategy,
        })

    def retrieve(self, task_description: str, top_k: int = 3) -> list[str]:
        return [s["corrective_strategy"] for s in self._strategies[-top_k:]]

    def count(self) -> int:
        return len(self._strategies)


class StrategyMemory:
    """Store and retrieve corrective strategies by similarity."""

    def __init__(
        self,
        persist_path: Path | str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_path = Path(persist_path) if persist_path else Path("chroma_db")
        self.embedding_model_name = embedding_model
        self._model = None
        self._client = None
        self._collection = None
        self._use_fallback = not (HAS_SENTENCE_TRANSFORMERS and HAS_CHROMADB)
        if self._use_fallback:
            self._fallback = _FallbackStrategyMemory(persist_path, embedding_model)
        else:
            self._fallback = None

    def _ensure_model(self):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers required for StrategyMemory")
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)

    def _ensure_client(self):
        if not HAS_CHROMADB:
            raise ImportError("chromadb required for StrategyMemory")
        if self._client is None:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_path),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name="strategies",
                metadata={"description": "Corrective strategies from failures"},
            )

    def add(
        self,
        task_id: str,
        task_description: str,
        failure_category: str,
        corrective_strategy: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a corrective strategy with embedding of task+strategy."""
        if self._use_fallback:
            self._fallback.add(task_id, task_description, failure_category,
                              corrective_strategy, metadata)
            return
        self._ensure_model()
        self._ensure_client()

        text = f"Task: {task_description}. Strategy: {corrective_strategy}"
        emb = self._model.encode(text).tolist()

        meta = {
            "task_id": task_id,
            "failure_category": failure_category,
            "corrective_strategy": corrective_strategy,
        }
        if metadata:
            meta.update(metadata)

        # ChromaDB needs string values for metadata
        meta = {k: str(v) for k, v in meta.items()}

        doc_id = f"{task_id}_{hash(corrective_strategy) % 10**8}"
        self._collection.upsert(
            ids=[doc_id],
            embeddings=[emb],
            documents=[text],
            metadatas=[meta],
        )

    def retrieve(
        self,
        task_description: str,
        top_k: int = 3,
    ) -> list[str]:
        """Retrieve top-k similar corrective strategies for a task."""
        if self._use_fallback:
            return self._fallback.retrieve(task_description, top_k)
        self._ensure_model()
        self._ensure_client()

        n = self._collection.count()
        if n == 0:
            return []

        emb = self._model.encode(task_description).tolist()
        results = self._collection.query(
            query_embeddings=[emb],
            n_results=min(top_k, n),
        )

        strategies = []
        if results and results.get("metadatas") and results["metadatas"][0]:
            for m in results["metadatas"][0]:
                s = m.get("corrective_strategy")
                if s:
                    strategies.append(s)
        return strategies

    def count(self) -> int:
        """Number of stored strategies."""
        if self._use_fallback:
            return self._fallback.count()
        self._ensure_client()
        return self._collection.count()
