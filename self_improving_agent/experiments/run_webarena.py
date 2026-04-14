"""
Full WebArena experiment script.

Usage:
    python -m self_improving_agent.experiments.run_webarena
    python -m self_improving_agent.experiments.run_webarena --dry-run
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
from self_improving_agent.agent.plan_act_agent import PlanActAgent
from self_improving_agent.agent.strategy_agent import StrategyAgent
from self_improving_agent.environments.web_env import generate_web_tasks
from self_improving_agent.evaluation.evaluate import run_experiment
from self_improving_agent.evaluation.metrics import (
    generate_summary_table,
    plot_cumulative_success,
    plot_failure_mode_dist,
    plot_success_vs_horizon,
)
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

    horizons = args.horizons
    n_tasks = args.n_tasks
    dry_run = args.dry_run

    console.rule("[bold blue]WebArena Experiment")
    console.print(f"Horizons: {horizons} | Tasks per horizon: {n_tasks} | Dry run: {dry_run}")

    llm_client = LLMClient(config)

    # ----------------------------------------------------------------
    # ReAct baseline
    # ----------------------------------------------------------------
    console.rule("[cyan]Condition: ReAct Baseline")
    react_tasks = generate_web_tasks(horizons=horizons, n_per_horizon=n_tasks)
    react_df = run_experiment(
        agent_class=BaseAgent,
        tasks=react_tasks,
        config=config,
        use_memory=False,
        llm_client=llm_client,
        label="react",
        dry_run=dry_run,
    )

    # ----------------------------------------------------------------
    # Plan-and-Act baseline
    # ----------------------------------------------------------------
    console.rule("[cyan]Condition: Plan-and-Act Baseline")
    plan_tasks = generate_web_tasks(horizons=horizons, n_per_horizon=n_tasks)
    plan_df = run_experiment(
        agent_class=PlanActAgent,
        tasks=plan_tasks,
        config=config,
        use_memory=False,
        llm_client=llm_client,
        label="plan_act",
        dry_run=dry_run,
    )

    # ----------------------------------------------------------------
    # Our method
    # ----------------------------------------------------------------
    console.rule("[cyan]Condition: Self-Improving (Our Method)")
    db_path = str(RESULTS_DIR / "strategy_memory_webarena.db")
    memory = StrategyMemory(db_path=db_path)
    retriever = Retriever(config=config, memory=memory)

    strategy_tasks = generate_web_tasks(horizons=horizons, n_per_horizon=n_tasks)
    strategy_df = run_experiment(
        agent_class=StrategyAgent,
        tasks=strategy_tasks,
        config=config,
        use_memory=True,
        memory=memory,
        retriever=retriever,
        llm_client=llm_client,
        label="self_improving",
        dry_run=dry_run,
    )

    # ----------------------------------------------------------------
    # Save and plot
    # ----------------------------------------------------------------
    results = {
        "ReAct": react_df,
        "Plan-and-Act": plan_df,
        "Self-Improving (ours)": strategy_df,
    }

    all_df = pd.concat(list(results.values()), ignore_index=True)
    csv_path = RESULTS_DIR / "webarena_results.csv"
    all_df.to_csv(csv_path, index=False)
    console.print(f"\n[green]Results saved to {csv_path}")

    plot_success_vs_horizon(results, save_path=str(RESULTS_DIR / "webarena_success_vs_horizon"))
    plot_failure_mode_dist(results, save_path=str(RESULTS_DIR / "webarena_failure_mode_dist"))
    plot_cumulative_success(results, save_path=str(RESULTS_DIR / "webarena_cumulative_success"))

    summary = generate_summary_table(results)
    summary.to_csv(RESULTS_DIR / "webarena_summary_table.csv", index=False)

    table = Table(title="WebArena Results")
    for col in summary.columns:
        table.add_column(str(col))
    for _, row in summary.iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WebArena experiment")
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 15, 20])
    parser.add_argument("--n-tasks", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
