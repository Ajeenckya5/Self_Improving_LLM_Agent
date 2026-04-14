"""
Ablation study: evaluates the contribution of each system component.

Conditions:
  1. Full system     — strategy agent + failure analysis + memory
  2. No memory       — analyze failures but don't store/retrieve
  3. No analysis     — store/retrieve but skip structured failure analysis
  4. No improvement  — plain ReAct baseline

Usage:
    python -m self_improving_agent.experiments.ablation
    python -m self_improving_agent.experiments.ablation --dry-run
    python -m self_improving_agent.experiments.ablation --horizon 15 --n-tasks 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from self_improving_agent.agent.base_agent import BaseAgent
from self_improving_agent.agent.strategy_agent import StrategyAgent
from self_improving_agent.environments.os_env import generate_os_tasks
from self_improving_agent.evaluation.evaluate import run_experiment
from self_improving_agent.evaluation.metrics import generate_summary_table, plot_ablation
from self_improving_agent.memory.retriever import Retriever
from self_improving_agent.memory.strategy_memory import StrategyMemory
from self_improving_agent.utils.llm_client import LLMClient

console = Console()
RESULTS_DIR = Path("results")


def main(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    horizon = args.horizon
    n_tasks = args.n_tasks
    dry_run = args.dry_run

    console.rule("[bold blue]Ablation Study")
    console.print(f"Horizon: {horizon} | Tasks: {n_tasks} | Dry run: {dry_run}")

    llm_client = LLMClient(config)

    # ----------------------------------------------------------------
    # Condition 1: Full system
    # ----------------------------------------------------------------
    console.rule("[cyan]Condition 1: Full System")
    full_memory = StrategyMemory(db_path=str(RESULTS_DIR / "ablation_full.db"))
    full_retriever = Retriever(config=config, memory=full_memory)
    full_tasks = generate_os_tasks(horizons=[horizon], n_per_horizon=n_tasks)
    full_df = run_experiment(
        agent_class=StrategyAgent,
        tasks=full_tasks,
        config=config,
        use_memory=True,
        memory=full_memory,
        retriever=full_retriever,
        llm_client=llm_client,
        label="full_system",
        dry_run=dry_run,
    )

    # ----------------------------------------------------------------
    # Condition 2: No strategy memory (analyze but don't store/retrieve)
    # ----------------------------------------------------------------
    console.rule("[cyan]Condition 2: No Strategy Memory")
    no_mem_tasks = generate_os_tasks(horizons=[horizon], n_per_horizon=n_tasks)
    no_mem_df = run_experiment(
        agent_class=BaseAgent,
        tasks=no_mem_tasks,
        config=config,
        use_memory=False,   # analyze but don't retrieve/store
        llm_client=llm_client,
        label="no_memory",
        dry_run=dry_run,
    )

    # ----------------------------------------------------------------
    # Condition 3: No failure analysis (random strategy text, still retrieves)
    # ----------------------------------------------------------------
    console.rule("[cyan]Condition 3: No Failure Analysis")
    rand_memory = StrategyMemory(db_path=str(RESULTS_DIR / "ablation_rand.db"))
    rand_retriever = Retriever(config=config, memory=rand_memory)

    # Pre-populate memory with dummy strategies to simulate retrieval without real analysis
    _seed_random_strategies(rand_memory, rand_retriever, n=10)

    rand_tasks = generate_os_tasks(horizons=[horizon], n_per_horizon=n_tasks)
    rand_df = run_experiment(
        agent_class=StrategyAgent,
        tasks=rand_tasks,
        config=config,
        use_memory=True,
        memory=rand_memory,
        retriever=rand_retriever,
        llm_client=llm_client,
        label="no_analysis",
        dry_run=dry_run,
    )

    # ----------------------------------------------------------------
    # Condition 4: Plain ReAct baseline (no improvement)
    # ----------------------------------------------------------------
    console.rule("[cyan]Condition 4: No Self-Improvement (ReAct Baseline)")
    react_tasks = generate_os_tasks(horizons=[horizon], n_per_horizon=n_tasks)
    react_df = run_experiment(
        agent_class=BaseAgent,
        tasks=react_tasks,
        config=config,
        use_memory=False,
        llm_client=llm_client,
        label="no_improvement",
        dry_run=dry_run,
    )

    # ----------------------------------------------------------------
    # Save and plot
    # ----------------------------------------------------------------
    results = {
        "Full System": full_df,
        "No Memory": no_mem_df,
        "No Analysis": rand_df,
        "No Self-Improvement": react_df,
    }

    all_df = pd.concat(list(results.values()), ignore_index=True)
    csv_path = RESULTS_DIR / "ablation_results.csv"
    all_df.to_csv(csv_path, index=False)
    console.print(f"\n[green]Results saved to {csv_path}")

    plot_ablation(results, save_path=str(RESULTS_DIR / "ablation"))
    console.print("[green]Ablation plot saved to results/ablation.png and results/ablation.pdf")

    summary = generate_summary_table(results)
    summary.to_csv(RESULTS_DIR / "ablation_summary.csv", index=False)

    table = Table(title="Ablation Study Results")
    for col in summary.columns:
        table.add_column(str(col))
    for _, row in summary.iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)


def _seed_random_strategies(memory: StrategyMemory, retriever: Retriever, n: int) -> None:
    """Seed memory with generic (non-analyzed) strategies for the no-analysis ablation."""
    generic_strategies = [
        "Try a different approach if the first attempt fails.",
        "Make sure to verify each step before proceeding.",
        "Read error messages carefully to understand what went wrong.",
        "Break complex tasks into smaller sub-tasks.",
        "Check file permissions before reading or writing.",
        "Use absolute paths to avoid directory confusion.",
        "Verify command output before assuming success.",
        "If stuck, try a simpler variation of the action.",
        "Confirm the current working directory before running commands.",
        "Double-check that required files exist before processing them.",
    ]

    import numpy as np
    for i in range(n):
        text = generic_strategies[i % len(generic_strategies)]
        try:
            emb = retriever.embed(text)
        except Exception:
            rng = np.random.default_rng(seed=i)
            emb = rng.standard_normal(384).astype(np.float32)

        memory.store(
            task_description=f"Generic task {i}",
            failure_analysis={"failure_type": "other", "tags": ["generic"]},
            strategy_text=text,
            embedding=emb,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--horizon", type=int, default=15, help="Task horizon (default: 15)")
    parser.add_argument("--n-tasks", type=int, default=25, help="Tasks per condition")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
