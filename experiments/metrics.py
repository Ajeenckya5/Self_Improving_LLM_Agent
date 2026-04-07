"""Metrics and plotting for experiments."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentMetrics:
    """Aggregated experiment metrics."""

    baseline_successes: int = 0
    baseline_total: int = 0
    baseline_steps: list[int] = field(default_factory=list)
    baseline_failures_by_task: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    baseline_failure_categories: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    strategy_successes: int = 0
    strategy_total: int = 0
    strategy_steps: list[int] = field(default_factory=list)
    strategy_failures_by_task: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(
        self,
        baseline_results: list[dict],
        strategy_results: list[dict],
    ) -> None:
        for r in baseline_results:
            self.baseline_total += 1
            if r["success"]:
                self.baseline_successes += 1
                if r.get("trace") and r["trace"].steps:
                    self.baseline_steps.append(len(r["trace"].steps))
            else:
                self.baseline_failures_by_task[r["task_id"]] += 1

        for r in strategy_results:
            self.strategy_total += 1
            if r["success"]:
                self.strategy_successes += 1
                if r.get("trace") and r["trace"].steps:
                    self.strategy_steps.append(len(r["trace"].steps))
            else:
                self.strategy_failures_by_task[r["task_id"]] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": {
                "success_rate": self.baseline_successes / max(1, self.baseline_total),
                "successes": self.baseline_successes,
                "total": self.baseline_total,
                "avg_steps_when_success": sum(self.baseline_steps) / max(1, len(self.baseline_steps)),
                "max_steps_solved": max(self.baseline_steps) if self.baseline_steps else 0,
            },
            "strategy_enhanced": {
                "success_rate": self.strategy_successes / max(1, self.strategy_total),
                "successes": self.strategy_successes,
                "total": self.strategy_total,
                "avg_steps_when_success": sum(self.strategy_steps) / max(1, len(self.strategy_steps)),
                "max_steps_solved": max(self.strategy_steps) if self.strategy_steps else 0,
            },
        }


def produce_plots(metrics: ExperimentMetrics, output_dir: str) -> list[str]:
    """Generate comparison plots. Returns paths to saved figures."""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. Success rate comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    agents = ["Baseline", "Strategy-Enhanced"]
    success_rates = [
        metrics.baseline_successes / max(1, metrics.baseline_total),
        metrics.strategy_successes / max(1, metrics.strategy_total),
    ]
    totals = [metrics.baseline_total, metrics.strategy_total]
    bars = ax.bar(agents, success_rates, color=["#e74c3c", "#2ecc71"])
    ax.set_ylabel("Task Success Rate")
    ax.set_ylim(0, 1.1)
    for bar, t in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={t}", ha="center", fontsize=10)
    plt.tight_layout()
    p1 = out / "success_rate.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    paths.append(str(p1))

    # 2. Max steps solved
    fig, ax = plt.subplots(figsize=(6, 4))
    max_steps_b = max(metrics.baseline_steps) if metrics.baseline_steps else 0
    max_steps_s = max(metrics.strategy_steps) if metrics.strategy_steps else 0
    ax.bar(["Baseline", "Strategy-Enhanced"], [max_steps_b, max_steps_s],
           color=["#e74c3c", "#2ecc71"])
    ax.set_ylabel("Max Steps (when successful)")
    plt.tight_layout()
    p2 = out / "max_steps.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    paths.append(str(p2))

    # 3. Failure distribution by task
    all_tasks = sorted(set(metrics.baseline_failures_by_task) | set(metrics.strategy_failures_by_task))
    if all_tasks:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(all_tasks))
        w = 0.35
        bl = [metrics.baseline_failures_by_task[t] for t in all_tasks]
        se = [metrics.strategy_failures_by_task[t] for t in all_tasks]
        ax.bar(x - w / 2, bl, w, label="Baseline", color="#e74c3c")
        ax.bar(x + w / 2, se, w, label="Strategy-Enhanced", color="#2ecc71")
        ax.set_xticks(x)
        ax.set_xticklabels(all_tasks, rotation=45, ha="right")
        ax.set_ylabel("Failures")
        ax.legend()
        plt.tight_layout()
        p3 = out / "failure_by_task.png"
        plt.savefig(p3, dpi=150)
        plt.close()
        paths.append(str(p3))

    return paths
