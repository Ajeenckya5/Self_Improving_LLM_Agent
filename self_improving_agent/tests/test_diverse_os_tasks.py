"""Tests for the large diverse OS task generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ..agent.strategy_agent import StrategyAgent
from ..environments.diverse_os_tasks import (
    generate_diverse_os_task_manifest,
    generate_diverse_os_tasks,
)
from ..utils.llm_client import LLMClient


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "model": {
            "primary": "gpt-4",
            "analyzer": "gpt-4",
            "embedding": "all-MiniLM-L6-v2",
        },
        "agent": {"max_steps": 4, "temperature": 0.0, "max_tokens": 500},
        "memory": {
            "top_k": 3,
            "similarity_threshold": 0.65,
            "db_path": str(tmp_path / "strategy_memory.db"),
        },
        "analysis": {"failure_analyzer": "heuristic"},
        "logging": {"llm_calls_log": str(tmp_path / "llm_calls.jsonl")},
    }


def test_diverse_manifest_has_5000_unique_tasks() -> None:
    manifest = generate_diverse_os_task_manifest(
        horizons=list(range(1, 51)),
        n_per_horizon=100,
    )

    assert len(manifest) == 5000
    assert {row["horizon"] for row in manifest} == set(range(1, 51))

    ids = [row["id"] for row in manifest]
    descriptions = [row["description"] for row in manifest]
    assert len(ids) == len(set(ids))
    assert len(descriptions) == len(set(descriptions))

    per_horizon = {horizon: 0 for horizon in range(1, 51)}
    for row in manifest:
        per_horizon[row["horizon"]] += 1
    assert set(per_horizon.values()) == {100}

    families = {row["family"] for row in manifest}
    assert families == {
        "os_cfg",
        "os_debug",
        "os_dirs",
        "os_find",
        "os_fulldbg",
        "os_git",
        "os_lines",
        "os_move",
        "os_multi",
        "os_perms",
        "os_pipeline",
        "os_sed",
    }


@pytest.mark.parametrize("horizon", [1, 50])
def test_mock_agent_solves_one_diverse_task_per_family(
    horizon: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOCK_LLM", "1")
    config = _config(tmp_path)
    llm = LLMClient(config)
    tasks = generate_diverse_os_tasks(horizons=[horizon], n_per_horizon=12)

    try:
        for task in tasks:
            agent = StrategyAgent(config=config, llm_client=llm, retriever=None)
            success, trace = agent.run(task, strategies=[])
            assert success is True, trace.to_text()
            assert trace.total_steps == 2
    finally:
        for task in tasks:
            env = task.get("env")
            if env is not None:
                env.cleanup()
