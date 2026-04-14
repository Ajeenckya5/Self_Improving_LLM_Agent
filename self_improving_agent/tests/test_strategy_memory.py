"""
Tests for StrategyMemory — store + retrieve, no API calls.
"""

import os
import tempfile

import numpy as np
import pytest

from ..memory.strategy_memory import StrategyMemory


@pytest.fixture
def tmp_memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_strategies.db")
        mem = StrategyMemory(db_path=db_path)
        yield mem


def _rand_emb(dim=384, seed=0):
    rng = np.random.default_rng(seed=seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestStore:
    def test_store_returns_id(self, tmp_memory):
        emb = _rand_emb()
        row_id = tmp_memory.store(
            task_description="Find a file",
            failure_analysis={"failure_type": "repeated_action", "tags": ["file", "search"]},
            strategy_text="Check whether the file exists before searching.",
            embedding=emb,
        )
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_count_increases(self, tmp_memory):
        assert tmp_memory.count() == 0
        emb = _rand_emb()
        tmp_memory.store("task", {"failure_type": "other", "tags": []}, "strategy", emb)
        assert tmp_memory.count() == 1
        tmp_memory.store("task2", {"failure_type": "other", "tags": []}, "strategy2", _rand_emb(seed=1))
        assert tmp_memory.count() == 2

    def test_tags_stored_as_json(self, tmp_memory):
        emb = _rand_emb()
        tmp_memory.store(
            "task", {"failure_type": "tool_misuse", "tags": ["tag1", "tag2"]}, "strat", emb
        )
        strategies = tmp_memory.all_strategies()
        assert len(strategies) == 1
        import json
        tags = json.loads(strategies[0]["tags"])
        assert "tag1" in tags


class TestRetrieve:
    def test_retrieve_above_threshold(self, tmp_memory):
        # Store a strategy with a known embedding
        emb_a = _rand_emb(seed=42)
        tmp_memory.store(
            "Find a file in a directory",
            {"failure_type": "repeated_action", "tags": ["file"]},
            "Use find command instead of ls.",
            emb_a,
        )

        # Query with the same embedding → should retrieve it
        results = tmp_memory.retrieve(emb_a, top_k=3, similarity_threshold=0.99)
        assert len(results) == 1
        assert results[0]["strategy_text"] == "Use find command instead of ls."

    def test_retrieve_below_threshold_returns_empty(self, tmp_memory):
        emb_a = _rand_emb(seed=1)
        emb_b = _rand_emb(seed=999)
        tmp_memory.store("task", {"failure_type": "other", "tags": []}, "strategy", emb_a)

        # Very high threshold + different embedding → no match
        results = tmp_memory.retrieve(emb_b, top_k=3, similarity_threshold=0.999)
        assert results == []

    def test_retrieve_top_k_limits(self, tmp_memory):
        # Store 5 similar strategies
        base_emb = _rand_emb(seed=0)
        for i in range(5):
            # Slightly perturb embedding
            noise = np.random.default_rng(seed=i+1).standard_normal(384).astype(np.float32) * 0.01
            emb = base_emb + noise
            emb /= np.linalg.norm(emb)
            tmp_memory.store(f"task {i}", {"failure_type": "other", "tags": []}, f"strategy {i}", emb)

        results = tmp_memory.retrieve(base_emb, top_k=3, similarity_threshold=0.5)
        assert len(results) <= 3

    def test_retrieve_increments_use_count(self, tmp_memory):
        emb = _rand_emb(seed=0)
        sid = tmp_memory.store("task", {"failure_type": "other", "tags": []}, "strat", emb)
        _ = tmp_memory.retrieve(emb, top_k=3, similarity_threshold=0.99)
        strategies = tmp_memory.all_strategies()
        assert strategies[0]["use_count"] == 1

    def test_retrieve_sorted_by_similarity(self, tmp_memory):
        base_emb = _rand_emb(seed=0)
        # Store two strategies, one closer to query
        close_emb = base_emb.copy()
        far_emb = _rand_emb(seed=42)

        tmp_memory.store("close task", {"failure_type": "other", "tags": []}, "close strategy", close_emb)
        tmp_memory.store("far task", {"failure_type": "other", "tags": []}, "far strategy", far_emb)

        results = tmp_memory.retrieve(base_emb, top_k=2, similarity_threshold=0.0)
        if len(results) >= 2:
            assert results[0]["similarity"] >= results[1]["similarity"]


class TestUpdateOutcome:
    def test_success_increments_success_count(self, tmp_memory):
        emb = _rand_emb()
        sid = tmp_memory.store("task", {"failure_type": "other", "tags": []}, "strat", emb)
        tmp_memory.update_outcome(sid, success=True)
        strategies = tmp_memory.all_strategies()
        assert strategies[0]["success_count"] == 1

    def test_failure_does_not_increment_success_count(self, tmp_memory):
        emb = _rand_emb()
        sid = tmp_memory.store("task", {"failure_type": "other", "tags": []}, "strat", emb)
        tmp_memory.update_outcome(sid, success=False)
        strategies = tmp_memory.all_strategies()
        assert strategies[0]["success_count"] == 0


class TestClear:
    def test_clear_empties_db(self, tmp_memory):
        emb = _rand_emb()
        tmp_memory.store("task", {"failure_type": "other", "tags": []}, "strat", emb)
        assert tmp_memory.count() == 1
        tmp_memory.clear()
        assert tmp_memory.count() == 0
