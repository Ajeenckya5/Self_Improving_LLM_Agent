"""
Synthetic results generator for offline development and presentation.

Produces plausible DataFrames that mirror what a real controlled experiment
would output, without requiring API keys. Uses fixed random seeds for
reproducibility.

Usage:
    from self_improving_agent.experiments.demo_results import generate_demo_results
    results = generate_demo_results()
"""

from __future__ import annotations

import random
from typing import Dict

import numpy as np
import pandas as pd

FAILURE_TYPES = [
    "repeated_action",
    "circular_loop",
    "context_truncation",
    "tool_misuse",
    "incorrect_reasoning",
]

TASK_IDS = [
    "fs_organize",
    "fs_nested",
    "fs_backup_clean",
    "db_schema",
    "db_insert",
    "db_query",
]

HORIZONS = [5, 10, 15, 20]

# Base success rates per condition per horizon (lower horizons = easier = higher success)
_SUCCESS_RATES: dict[str, list[float]] = {
    "ReAct":                    [0.55, 0.45, 0.35, 0.28],
    "Plan-and-Act":             [0.65, 0.55, 0.45, 0.38],
    "Self-Improving (ours)":    [0.72, 0.68, 0.62, 0.57],
}

# Failure type distributions per condition
_FAILURE_DIST: dict[str, list[float]] = {
    "ReAct":                 [0.35, 0.25, 0.20, 0.12, 0.08],
    "Plan-and-Act":          [0.25, 0.20, 0.25, 0.15, 0.15],
    "Self-Improving (ours)": [0.15, 0.15, 0.20, 0.25, 0.25],
}


def generate_demo_results(
    seed: int = 42,
    n_attempts: int = 3,
    results_dir: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic experiment results.

    Parameters
    ----------
    seed        : Random seed for reproducibility.
    n_attempts  : Number of repeated attempts per task per condition.
    results_dir : If given, saves CSVs there.

    Returns
    -------
    dict mapping condition label → DataFrame with columns:
        task_id, horizon, agent_type, attempt, success,
        steps_taken, failure_type, strategies_used, elapsed_s, label
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    conditions = list(_SUCCESS_RATES.keys())
    all_results: Dict[str, pd.DataFrame] = {}

    for label in conditions:
        rows = []
        success_rates = _SUCCESS_RATES[label]
        fail_probs = _FAILURE_DIST[label]
        agent_type = _label_to_agent_type(label)

        # Track seen failure types to model recurrence behaviour
        seen_failures: list[str] = []

        for attempt_idx in range(n_attempts):
            for task_id in TASK_IDS:
                for h_idx, horizon in enumerate(HORIZONS):
                    base_rate = success_rates[h_idx]

                    # Self-improving agent improves with more attempts
                    if label == "Self-Improving (ours)":
                        boost = 0.04 * attempt_idx
                    else:
                        boost = 0.0

                    success_prob = min(base_rate + boost, 0.95)
                    success = bool(rng.random() < success_prob)

                    failure_type = None
                    strategies_used = 0

                    if not success:
                        ft_idx = int(rng.choice(len(FAILURE_TYPES), p=fail_probs))
                        failure_type = FAILURE_TYPES[ft_idx]
                        seen_failures.append(failure_type)
                    else:
                        if label == "Self-Improving (ours)" and attempt_idx > 0:
                            strategies_used = int(rng.integers(1, 4))

                    max_steps = horizon
                    steps_taken = (
                        int(rng.integers(1, max(2, max_steps // 2))) if success
                        else max_steps
                    )

                    rows.append({
                        "task_id": task_id,
                        "horizon": horizon,
                        "agent_type": agent_type,
                        "attempt": attempt_idx,
                        "success": success,
                        "steps_taken": steps_taken,
                        "failure_type": failure_type,
                        "strategies_used": strategies_used,
                        "elapsed_s": round(float(rng.uniform(0.5, 8.0)), 2),
                        "label": label,
                    })

        df = pd.DataFrame(rows)
        all_results[label] = df

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
