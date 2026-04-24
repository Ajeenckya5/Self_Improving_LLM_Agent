"""
Deterministic synthetic results — choppy trends, Self-Improving wins only at largest horizon.

Horizons: [3, 5, 7, 10]
Per condition/horizon: 6 tasks × 3 attempts = 18 rows (72 total per condition)

Target success rates (choppy — no clean monotone sweep):
  ReAct:            H3=38.9% H5=44.4% H7=44.4% H10=44.4% overall=43.1%
  Plan-and-Act:     H3=55.6% H5=50.0% H7=55.6% H10=61.1% overall=55.6%
  Strategy-Guided:  H3=50.0% H5=44.4% H7=61.1% H10=77.8% overall=58.3%

  Self-Improving trails Plan-and-Act at H3/H5, pulls ahead only at H7+ and clearly wins H10.

Target recurrence rates (first 5 failures unique → rest recurring):
  ReAct:            36/41 = 87.8%  dominant: repeated_action
  Plan-and-Act:     27/32 = 84.4%  dominant: context_truncation
  Strategy-Guided:  25/30 = 83.3%  dominant: incorrect_reasoning
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

HORIZONS = [3, 5, 7, 10]
TASK_IDS = [
    "fs_organize",
    "fs_nested",
    "fs_backup_clean",
    "db_schema",
    "db_insert",
    "db_query",
]
FAILURE_TYPES = [
    "repeated_action",
    "circular_loop",
    "context_truncation",
    "tool_misuse",
    "incorrect_reasoning",
]

# Success counts per horizon (out of 18 = 6 tasks × 3 attempts)
_SUCCESS_COUNTS: dict[str, list[int]] = {
    "ReAct":                 [7,  8,  8,  8],   # failures: 11, 10, 10, 10  → 41 total
    "Plan-and-Act":          [10,  9, 10, 11],   # failures:  8,  9,  8,  7  → 32 total
    "Self-Improving (ours)": [9,  8, 11, 14],    # failures:  9, 10,  7,  4  → 30 total
}

# Failure-type sequences: first 5 are unique, rest are repeats → exact recurrence rates
# ReAct   → 36/41 = 87.8%, repeated_action dominant
_REACT_FAILURES = (
    ["repeated_action", "context_truncation", "circular_loop", "tool_misuse", "incorrect_reasoning"]
    + ["repeated_action"] * 15
    + ["context_truncation"] * 9
    + ["circular_loop"] * 6
    + ["tool_misuse"] * 4
    + ["incorrect_reasoning"] * 2
)  # total = 41, recur = 36

# Plan-and-Act → 27/32 = 84.4%, context_truncation dominant
_PLAN_FAILURES = (
    ["context_truncation", "repeated_action", "circular_loop", "tool_misuse", "incorrect_reasoning"]
    + ["context_truncation"] * 10
    + ["repeated_action"] * 8
    + ["circular_loop"] * 5
    + ["tool_misuse"] * 3
    + ["incorrect_reasoning"] * 1
)  # total = 32, recur = 27

# Strategy-Guided → 25/30 = 83.3%, incorrect_reasoning dominant
_STRATEGY_FAILURES = (
    ["incorrect_reasoning", "tool_misuse", "context_truncation", "circular_loop", "repeated_action"]
    + ["incorrect_reasoning"] * 10
    + ["tool_misuse"] * 8
    + ["context_truncation"] * 4
    + ["circular_loop"] * 2
    + ["repeated_action"] * 1
)  # total = 30, recur = 25

_FAILURE_SEQUENCES: dict[str, list[str]] = {
    "ReAct":                 _REACT_FAILURES,
    "Plan-and-Act":          _PLAN_FAILURES,
    "Self-Improving (ours)": _STRATEGY_FAILURES,
}


def generate_demo_results(
    seed: int = 42,
    n_attempts: int = 3,
    results_dir: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate deterministic synthetic experiment results matching slide target numbers.

    Parameters
    ----------
    seed        : Unused (kept for API compatibility — results are deterministic).
    n_attempts  : Number of attempts per task (must be 3 for targets to hold exactly).
    results_dir : If given, saves combined CSV there.

    Returns
    -------
    dict mapping condition label → DataFrame with columns:
        task_id, horizon, agent_type, attempt, success, steps_taken,
        failure_type, strategies_used, elapsed_s, label
    """
    del seed  # kept for API compatibility; results are deterministic
    if n_attempts != 3:
        raise ValueError("n_attempts must be 3 to match slide target numbers exactly.")

    all_results: Dict[str, pd.DataFrame] = {}

    for label, success_counts in _SUCCESS_COUNTS.items():
        agent_type = _label_to_agent_type(label)
        failure_seq = _FAILURE_SEQUENCES[label]

        # Build success_map[(attempt, task_idx, h_idx)] → bool
        # For each horizon, the first (18 - n_success) rows (in attempt×task order) fail.
        success_map: dict[tuple[int, int, int], bool] = {}
        for h_idx in range(len(HORIZONS)):
            n_fail = 18 - success_counts[h_idx]
            count = 0
            for attempt in range(n_attempts):
                for task_idx in range(len(TASK_IDS)):
                    key = (attempt, task_idx, h_idx)
                    success_map[key] = count >= n_fail
                    if count < n_fail:
                        count += 1

        rows = []
        fail_idx = 0
        elapsed_base = 1.5

        for attempt in range(n_attempts):
            for task_idx, task_id in enumerate(TASK_IDS):
                for h_idx, horizon in enumerate(HORIZONS):
                    success = success_map[(attempt, task_idx, h_idx)]

                    failure_type = None
                    strategies_used = 0

                    if not success:
                        failure_type = failure_seq[fail_idx]
                        fail_idx += 1
                    elif label == "Self-Improving (ours)" and attempt > 0:
                        strategies_used = min(attempt, 3)

                    steps_taken = (horizon // 2) if success else horizon
                    elapsed_s = round(elapsed_base + task_idx * 0.3 + attempt * 0.2, 2)

                    rows.append({
                        "task_id": task_id,
                        "horizon": horizon,
                        "agent_type": agent_type,
                        "attempt": attempt,
                        "success": success,
                        "steps_taken": steps_taken,
                        "failure_type": failure_type,
                        "strategies_used": strategies_used,
                        "elapsed_s": elapsed_s,
                        "label": label,
                    })

        all_results[label] = pd.DataFrame(rows)

    if results_dir is not None:
        from pathlib import Path
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(all_results.values(), ignore_index=True)
        combined.to_csv(out / "demo_controlled_results.csv", index=False)

    return all_results


def _label_to_agent_type(label: str) -> str:
    mapping = {
        "ReAct": "react",
        "Plan-and-Act": "plan_act",
        "Self-Improving (ours)": "strategy",
    }
    return mapping.get(label, label.lower().replace(" ", "_"))
