"""Prompt ablation study for Member 2 components.

Compares grounded failure analysis, weakened failure prompts, generic
strategy prompts, and heuristic-only analysis using the same judge harness.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from rich.console import Console
from rich.table import Table

from self_improving_agent.experiments.member2_eval import (
    DEFAULT_DATASET,
    load_config,
    run_member2_eval,
)

console = Console()
DEFAULT_OUTPUT_DIR = Path("results/member2_ablation")


def run_prompt_ablation(
    dataset_path: Path,
    output_dir: Path,
    config: Dict[str, Any],
) -> pd.DataFrame:
    conditions = config.get("analysis", {}).get("prompt_ablation_conditions", {})
    if not conditions:
        raise ValueError("No analysis.prompt_ablation_conditions found in config.yaml")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, condition in conditions.items():
        console.rule(f"[cyan]Member 2 Ablation: {name}")
        df = run_member2_eval(
            dataset_path=dataset_path,
            output_path=output_dir / f"{name}.csv",
            config=config,
            analyzer_mode=condition.get("analyzer", "llm"),
            failure_prompt=condition.get("failure_prompt"),
            strategy_prompt=condition.get("strategy_prompt"),
        )
        df.insert(0, "condition", name)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_dir / "member2_ablation_results.csv"
    combined.to_csv(combined_path, index=False)

    summary = summarize(combined)
    summary.to_csv(output_dir / "member2_ablation_summary.csv", index=False)
    print_summary(summary)
    console.print(f"[green]Wrote {combined_path}")
    return combined


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("condition", sort=False)
    return grouped.agg(
        n=("task_id", "count"),
        type_accuracy=("failure_type_correct", "mean"),
        step_overlap=("failed_steps_overlap", "mean"),
        grounding=("analysis_grounding_score", "mean"),
        specificity=("strategy_specificity_score", "mean"),
        actionability=("strategy_actionability_score", "mean"),
        retrieval_tags=("retrieval_tags_score", "mean"),
        overall=("overall_score", "mean"),
    ).reset_index().round(3)


def print_summary(summary: pd.DataFrame) -> None:
    table = Table(title="Member 2 Prompt Ablation Summary")
    for col in summary.columns:
        table.add_column(str(col))
    for _, row in summary.iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Member 2 prompt ablation study")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile", default=None, help="Model profile: xai | haiku | groq | ollama")
    parser.add_argument("--mock", action="store_true", help="Force deterministic mock LLM")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.mock:
        os.environ["MOCK_LLM"] = "1"
    config = load_config(args.profile)
    run_prompt_ablation(args.dataset, args.output_dir, config)


if __name__ == "__main__":
    main(parse_args())
