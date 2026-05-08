"""
Evaluation metrics and plot generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
STYLE = "seaborn-v0_8-whitegrid"

try:
    plt.style.use(STYLE)
except OSError:
    pass  # older matplotlib


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def task_success_rate(df: pd.DataFrame, groupby: Optional[str] = None) -> pd.Series:
    """Overall task success rate, optionally broken down by a column."""
    if groupby:
        return df.groupby(groupby)["success"].mean()
    return pd.Series({"overall": df["success"].mean()})


def failure_mode_distribution(df: pd.DataFrame) -> pd.Series:
    """Distribution of failure types among failed tasks."""
    failed = df[~df["success"]]
    if failed.empty:
        return pd.Series(dtype=float)
    return failed["failure_type"].value_counts(normalize=True)


def success_vs_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """Success rate by horizon for each method/agent_type."""
    if "agent_type" not in df.columns:
        return df.groupby("horizon")["success"].mean().reset_index()
    return (
        df.groupby(["horizon", "agent_type"])["success"]
        .mean()
        .reset_index()
        .rename(columns={"success": "success_rate"})
    )


def cumulative_success_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative success rate over the task sequence."""
    if "agent_type" not in df.columns:
        df = df.copy()
        df["agent_type"] = "unknown"

    records = []
    for agent_type, group in df.groupby("agent_type"):
        group = group.reset_index(drop=True)
        cumsum = group["success"].cumsum()
        cumrate = cumsum / (group.index + 1)
        for i, (rate, task_n) in enumerate(zip(cumrate, group.index + 1)):
            records.append({"task_n": task_n, "agent_type": agent_type, "cumulative_success": rate})

    return pd.DataFrame(records)


def repeated_failure_rate(df: pd.DataFrame) -> float:
    """
    Fraction of tasks that failed with a failure type that already appeared
    in a prior failed task (i.e., the agent repeated the same mistake).
    """
    failed = df[~df["success"]].copy()
    if failed.empty:
        return 0.0

    seen: set = set()
    repeated = 0
    for _, row in failed.iterrows():
        ft = row.get("failure_type", "other")
        if ft in seen:
            repeated += 1
        seen.add(ft)

    return repeated / len(failed)


def generate_summary_table(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Given a dict of {condition_name: DataFrame}, compute mean±std per condition.
    """
    rows = []
    for condition, df in results.items():
        sr = df["success"].mean()
        sr_std = df["success"].std()
        rfr = repeated_failure_rate(df)
        rows.append({
            "condition": condition,
            "success_rate_mean": round(sr, 4),
            "success_rate_std": round(sr_std, 4),
            "repeated_failure_rate": round(rfr, 4),
            "n_tasks": len(df),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_success_vs_horizon(
    results: Dict[str, pd.DataFrame],
    save_path: str = "results/success_vs_horizon",
) -> None:
    """Line chart of success rate vs. horizon (or by task when single horizon)."""
    all_horizons: set = set()
    for df in results.values():
        all_horizons.update(df["horizon"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))

    if len(all_horizons) <= 1:
        # Single horizon — fall back to per-task success rate
        task_ids: list | None = None
        for condition, df in results.items():
            task_sr = df.groupby("task_id")["success"].mean().reset_index()
            task_sr = task_sr.sort_values("task_id").reset_index(drop=True)
            if task_ids is None:
                task_ids = list(task_sr["task_id"])
            ax.plot(range(len(task_sr)), task_sr["success"], marker="o", label=condition)
        ax.set_xticks(range(len(task_ids or [])))
        ax.set_xticklabels(task_ids or [], rotation=30, ha="right")
        ax.set_xlabel("Task")
        ax.set_title("Task Success Rate by Task")
    else:
        for condition, df in results.items():
            svh = success_vs_horizon(df)
            if "agent_type" in svh.columns:
                svh = svh.groupby("horizon")["success_rate"].mean().reset_index()
            elif "success" in svh.columns:
                svh = svh.rename(columns={"success": "success_rate"})
            ax.plot(svh["horizon"], svh["success_rate"], marker="o", label=condition)
        ax.set_xlabel("Task Horizon (steps)")
        ax.set_title("Task Success Rate vs. Horizon")

    ax.set_ylabel("Success Rate")
    ax.legend()
    ax.set_ylim(0, 1.05)
    _save_fig(fig, save_path)


def plot_failure_mode_dist(
    results: Dict[str, pd.DataFrame],
    save_path: str = "results/failure_mode_dist",
) -> None:
    """Grouped bar chart of failure type distributions per method."""
    all_types = set()
    for df in results.values():
        all_types.update(df[~df["success"]]["failure_type"].dropna().unique())

    if not all_types:
        return

    conditions = list(results.keys())
    failure_types = sorted(all_types)
    x = np.arange(len(failure_types))
    width = 0.8 / max(len(conditions), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (condition, df) in enumerate(results.items()):
        dist = failure_mode_distribution(df)
        vals = [dist.get(ft, 0.0) for ft in failure_types]
        ax.bar(x + i * width, vals, width, label=condition)

    ax.set_xticks(x + width * (len(conditions) - 1) / 2)
    ax.set_xticklabels(failure_types, rotation=30, ha="right")
    ax.set_ylabel("Fraction of Failures")
    ax.set_title("Failure Mode Distribution by Method")
    ax.legend()
    _save_fig(fig, save_path)


def plot_cumulative_success(
    results: Dict[str, pd.DataFrame],
    save_path: str = "results/cumulative_success",
) -> None:
    """Line chart of cumulative success rate over the task sequence."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for condition, df in results.items():
        if df is None or len(df) == 0 or "success" not in df.columns:
            continue
        # Sort consistently when attempt/task_id are present; otherwise keep insertion order.
        sort_keys = [c for c in ("attempt", "task_id") if c in df.columns]
        sorted_df = df.sort_values(sort_keys).reset_index(drop=True) if sort_keys else df.reset_index(drop=True)
        cumrate = sorted_df["success"].cumsum() / (sorted_df.index + 1)
        ax.plot(sorted_df.index + 1, cumrate.values, label=condition)

    ax.set_xlabel("Task Number")
    ax.set_ylabel("Cumulative Success Rate")
    ax.set_title("Learning Curve: Cumulative Success Over Task Sequence")
    ax.legend()
    ax.set_ylim(0, 1.05)
    _save_fig(fig, save_path)


def plot_ablation(
    results: Dict[str, pd.DataFrame],
    save_path: str = "results/ablation",
) -> None:
    """Bar chart comparing success rates across ablation conditions."""
    conditions = list(results.keys())
    rates = [df["success"].mean() for df in results.values()]
    errs = [df["success"].std() for df in results.values()]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(conditions))
    ax.bar(x, rates, yerr=errs, capsize=5, color="steelblue", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("Success Rate")
    ax.set_title("Ablation Study: Component Contribution")
    ax.set_ylim(0, 1.05)
    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, base_path: str) -> None:
    p = Path(base_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{p}.png", dpi=150)
    fig.savefig(f"{p}.pdf")
    plt.close(fig)
