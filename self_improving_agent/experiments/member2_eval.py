"""Member 2 evaluation harness for failure analysis and strategy prompts.

Usage:
    python -m self_improving_agent.experiments.member2_eval --mock
    python -m self_improving_agent.experiments.member2_eval --profile xai
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from self_improving_agent.analysis.failure_analyzer import FailureAnalyzer
from self_improving_agent.analysis.llm_failure_analyzer import LLMFailureAnalyzer
from self_improving_agent.analysis.strategy_generator import StrategyGenerator
from self_improving_agent.evaluation.llm_judge import FailureStrategyJudge
from self_improving_agent.utils.llm_client import LLMClient

console = Console()
DEFAULT_DATASET = Path("self_improving_agent/data/member2_eval_traces.jsonl")
DEFAULT_OUTPUT = Path("results/member2_eval_results.csv")


def run_member2_eval(
    dataset_path: Path,
    output_path: Path,
    config: Dict[str, Any],
    analyzer_mode: str = "llm",
    failure_prompt: str | None = None,
    strategy_prompt: str | None = None,
) -> pd.DataFrame:
    """Run analyzer, strategy generator, and judge over labeled traces."""
    records = list(_load_jsonl(dataset_path))
    llm_client = LLMClient(config)
    heuristic_analyzer = FailureAnalyzer()
    llm_analyzer = LLMFailureAnalyzer(
        config=config,
        llm_client=llm_client,
        prompt_name=failure_prompt,
        fallback=heuristic_analyzer,
    )
    generator = StrategyGenerator(
        config=config,
        llm_client=llm_client,
        prompt_name=strategy_prompt,
    )
    judge = FailureStrategyJudge(config=config, llm_client=llm_client)

    rows: List[Dict[str, Any]] = []
    details_path = output_path.with_suffix(".jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if details_path.exists():
        details_path.unlink()

    for record in records:
        task = {
            "id": record["id"],
            "description": record["task_description"],
            "gold_failure_type": record.get("gold_failure_type"),
            "gold_failed_steps": record.get("gold_failed_steps", []),
        }
        trace = {"steps": record["steps"]}

        if analyzer_mode == "heuristic":
            failure = heuristic_analyzer.analyze(trace)
            failure["analysis_source"] = "heuristic"
        else:
            failure = llm_analyzer.analyze(task, trace)

        strategy = generator.generate(task, failure)
        scores = judge.evaluate(task, trace, failure, strategy)

        row = {
            "task_id": task["id"],
            "analyzer_mode": analyzer_mode,
            "failure_prompt": failure_prompt or config.get("analysis", {}).get("failure_prompt"),
            "strategy_prompt": strategy_prompt or config.get("analysis", {}).get("strategy_prompt"),
            "gold_failure_type": task["gold_failure_type"],
            "pred_failure_type": failure.get("failure_type"),
            **scores,
        }
        rows.append(row)

        with open(details_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "task": task,
                "failure_analysis": failure,
                "strategy": strategy,
                "scores": scores,
            }) + "\n")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def load_config(profile: str | None = None) -> Dict[str, Any]:
    config_path = Path("self_improving_agent/config.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    active = profile or config.get("active_profile", "xai")
    profiles = config.get("model_profiles", {})
    if active in profiles:
        config["model"].update(profiles[active])
    return config


def print_summary(df: pd.DataFrame) -> None:
    summary = {
        "rows": len(df),
        "type_accuracy": round(float(df["failure_type_correct"].mean()), 3) if len(df) else 0.0,
        "avg_step_overlap": round(float(df["failed_steps_overlap"].mean()), 3) if len(df) else 0.0,
        "avg_overall_score": round(float(df["overall_score"].mean()), 3) if len(df) else 0.0,
    }
    table = Table(title="Member 2 Evaluation")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in summary.items():
        table.add_row(key, str(value))
    console.print(table)


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Member 2 prompt evaluation harness")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", default=None, help="Model profile: xai | haiku | groq | ollama")
    parser.add_argument("--mock", action="store_true", help="Force deterministic mock LLM")
    parser.add_argument("--analyzer-mode", choices=["llm", "heuristic"], default="llm")
    parser.add_argument("--failure-prompt", default=None)
    parser.add_argument("--strategy-prompt", default=None)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.mock:
        os.environ["MOCK_LLM"] = "1"
    config = load_config(args.profile)
    df = run_member2_eval(
        dataset_path=args.dataset,
        output_path=args.output,
        config=config,
        analyzer_mode=args.analyzer_mode,
        failure_prompt=args.failure_prompt,
        strategy_prompt=args.strategy_prompt,
    )
    console.print(f"[green]Wrote {args.output} and {args.output.with_suffix('.jsonl')}")
    print_summary(df)


if __name__ == "__main__":
    main(parse_args())
