"""
Failure mode recurrence analysis.

Measures whether agents repeat the same mistake types across the task sequence.
A self-improving agent should show decreasing recurrence as it accumulates memory.

Key functions:
    failure_recurrence_over_time(df)          -> rolling recurrence rate per task index
    compare_recurrence_across_conditions(...) -> summary table per condition
    plot_failure_recurrence(...)               -> line chart saved to disk
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass


def failure_recurrence_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative failure recurrence rate over the task sequence.

    A failure is "recurring" if its `failure_type` was already seen in an
    earlier failed task within the same condition run.

    Parameters
    ----------
    df : DataFrame for a single condition, ordered by task sequence.
         Must have columns: success (bool), failure_type (str | None).

    Returns
    -------
    DataFrame with columns: task_n, recurrence_rate
    """
    failed = df[~df["success"]].reset_index(drop=True)
    if failed.empty:
        return pd.DataFrame({"task_n": [], "recurrence_rate": []})

    seen: set[str] = set()
    is_recurring_list: list[bool] = []

    for _, row in failed.iterrows():
        ft = str(row.get("failure_type") or "other")
        is_recurring_list.append(ft in seen)
        seen.add(ft)

    # Build cumulative rate
    result = []
    running_recurring = 0
    for idx, recurring in enumerate(is_recurring_list):
        running_recurring += int(recurring)
        result.append({
            "task_n": idx + 1,
            "recurrence_rate": running_recurring / (idx + 1),
        })

    return pd.DataFrame(result)


def compare_recurrence_across_conditions(
    results: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Summary table: one row per condition with failure recurrence statistics.

    Columns: condition, n_tasks, n_failures, n_unique_failure_types,
             n_recurring_failures, recurrence_rate, top_failure_type
    """
    rows = []
    for condition, df in results.items():
        failed = df[~df["success"]]
        n_tasks = len(df)
        n_failures = len(failed)

        if failed.empty:
            rows.append({
                "condition": condition,
                "n_tasks": n_tasks,
                "n_failures": 0,
                "n_unique_failure_types": 0,
                "n_recurring_failures": 0,
                "recurrence_rate": 0.0,
                "top_failure_type": "—",
            })
            continue

        ft_series = failed["failure_type"].dropna()  # type: ignore[assignment]
        failure_types = ft_series.tolist()
        unique_types = set(failure_types)

        seen: set[str] = set()
        recurring = 0
        for ft in failure_types:
            if ft in seen:
                recurring += 1
            seen.add(ft)

        top_type = ft_series.value_counts().idxmax() if not ft_series.empty else "—"

        rows.append({
            "condition": condition,
            "n_tasks": n_tasks,
            "n_failures": n_failures,
            "n_unique_failure_types": len(unique_types),
            "n_recurring_failures": recurring,
            "recurrence_rate": round(recurring / max(n_failures, 1), 4),
            "top_failure_type": top_type,
        })

    return pd.DataFrame(rows)


def plot_failure_recurrence(
    results: Dict[str, pd.DataFrame],
    save_path: str = "results/failure_recurrence",
) -> None:
    """
    Line chart: cumulative failure recurrence rate vs. number of failed tasks seen,
    one line per condition.

    A downward slope (or flat line at lower rate) for the self-improving agent
    indicates the memory is preventing repeated mistakes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for condition, df in results.items():
        curve = failure_recurrence_over_time(df)
        if curve.empty:
            continue
        ax.plot(curve["task_n"], curve["recurrence_rate"], marker="o", label=condition)

    ax.set_xlabel("Failed Task Index")
    ax.set_ylabel("Cumulative Failure Recurrence Rate")
    ax.set_title("Failure Mode Recurrence Over Task Sequence")
    ax.legend()
    ax.set_ylim(0, 1.05)

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{p}.png", dpi=150)
    fig.savefig(f"{p}.pdf")
    plt.close(fig)


def print_recurrence_summary(results: Dict[str, pd.DataFrame]) -> None:
    """Print a formatted recurrence comparison table to stdout."""
    table = compare_recurrence_across_conditions(results)
    print("\n--- Failure Mode Recurrence Analysis ---")
    print(table.to_string(index=False))

    # Highlight reduction vs worst baseline
    if len(table) >= 2:
        baseline_max = table[table["condition"] != "Self-Improving (ours)"]["recurrence_rate"].max()
        ours_row = table[table["condition"] == "Self-Improving (ours)"]
        if not ours_row.empty:
            ours_rate = ours_row.iloc[0]["recurrence_rate"]
            reduction = (baseline_max - ours_rate) / max(baseline_max, 1e-9) * 100
            print(
                f"\nSelf-Improving agent reduces failure recurrence by "
                f"{reduction:.1f}% vs. best baseline."
            )
