"""
Self-Improving LLM Agent — unified entry point.

Commands:
  python main.py run             Run controlled task experiment (filesystem + database)
  python main.py agentbench      Run AgentBench OS experiment
  python main.py ablation        Run ablation study

Set OPENAI_API_KEY (or ANTHROPIC_API_KEY) in .env or environment.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env if present
# ---------------------------------------------------------------------------
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

sys.path.insert(0, str(Path(__file__).parent))

import yaml


def _load_config() -> dict:
    config_path = Path(__file__).parent / "self_improving_agent" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Run controlled filesystem + database task experiment."""
    from self_improving_agent.experiments.controlled_runner import run_controlled_experiment
    from self_improving_agent.evaluation.metrics import (
        generate_summary_table,
        plot_success_vs_horizon,
        plot_failure_mode_dist,
        plot_cumulative_success,
    )

    config = _load_config()
    results_dir = args.output

    print("Running controlled task experiment (filesystem + database tasks)...")
    results = run_controlled_experiment(
        config=config,
        env_root=args.sandbox_dir or "sandbox",
        results_dir=results_dir,
        num_attempts=args.attempts,
        dry_run=args.dry_run,
    )

    # Save CSV
    import pandas as pd
    all_df = pd.concat(list(results.values()), ignore_index=True)
    csv_path = Path(results_dir) / "controlled_results.csv"
    all_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Summary table
    summary = generate_summary_table(results)
    print("\n--- Summary ---")
    print(summary.to_string(index=False))
    summary_path = Path(results_dir) / "controlled_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Plots
    if not args.no_plot:
        plot_success_vs_horizon(results, save_path=str(Path(results_dir) / "controlled_success_vs_horizon"))
        plot_failure_mode_dist(results, save_path=str(Path(results_dir) / "controlled_failure_dist"))
        plot_cumulative_success(results, save_path=str(Path(results_dir) / "controlled_cumulative"))
        print(f"Plots saved to {results_dir}/")


def cmd_agentbench(args: argparse.Namespace) -> None:
    """Run AgentBench OS experiment."""
    import subprocess
    cmd = [sys.executable, "-m", "self_improving_agent.experiments.run_agentbench"]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.horizons:
        cmd += ["--horizons"] + [str(h) for h in args.horizons]
    if args.n_tasks:
        cmd += ["--n-tasks", str(args.n_tasks)]
    subprocess.run(cmd, check=True)


def cmd_ablation(args: argparse.Namespace) -> None:
    """Run ablation study."""
    import subprocess
    cmd = [sys.executable, "-m", "self_improving_agent.experiments.ablation"]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.horizon:
        cmd += ["--horizon", str(args.horizon)]
    if args.n_tasks:
        cmd += ["--n-tasks", str(args.n_tasks)]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-Improving LLM Agent evaluation framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", help="Command to run")

    # --- run ---
    run_p = sub.add_parser("run", help="Run controlled task experiment (filesystem + database)")
    run_p.add_argument("--attempts", type=int, default=1, help="Attempts per task per agent")
    run_p.add_argument("--output", "-o", default="results", help="Output directory for results/plots")
    run_p.add_argument("--sandbox-dir", default="sandbox", help="Sandbox directory for task environments")
    run_p.add_argument("--dry-run", action="store_true", help="Only run first 3 tasks (for testing)")
    run_p.add_argument("--no-plot", action="store_true", help="Skip generating plots")

    # --- agentbench ---
    ab_p = sub.add_parser("agentbench", help="Run AgentBench OS experiment")
    ab_p.add_argument("--horizons", nargs="+", type=int, default=None)
    ab_p.add_argument("--n-tasks", type=int, default=None)
    ab_p.add_argument("--dry-run", action="store_true")

    # --- ablation ---
    abl_p = sub.add_parser("ablation", help="Run ablation study")
    abl_p.add_argument("--horizon", type=int, default=None)
    abl_p.add_argument("--n-tasks", type=int, default=None)
    abl_p.add_argument("--dry-run", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "agentbench":
        cmd_agentbench(args)
    elif args.cmd == "ablation":
        cmd_ablation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
