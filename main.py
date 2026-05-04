"""
Self-Improving LLM Agent — unified entry point.

Commands:
  python main.py run             Run controlled task experiment (filesystem + database)
  python main.py agentbench      Run AgentBench OS experiment
  python main.py terminalbench   Run Terminal-Bench with the custom agent adapter
  python main.py ablation        Run ablation study
  python main.py member2-eval    Evaluate Member 2 failure/strategy prompts
  python main.py member2-ablation Run Member 2 prompt ablations

Set XAI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env or environment.
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


def _load_config(profile: "str | None" = None) -> dict:
    config_path = Path(__file__).parent / "self_improving_agent" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Apply model profile: CLI --profile > config active_profile > default
    active = profile or config.get("active_profile", "haiku")
    profiles = config.get("model_profiles", {})
    if active in profiles:
        config["model"].update(profiles[active])
    return config


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

    profile = getattr(args, "profile", None)
    config = _load_config(profile)
    print(f"Model profile: {profile or config.get('active_profile', 'haiku')} "
          f"({config['model']['primary']} / {config['model']['backend']})")
    results_dir = args.output

    print("Running controlled task experiment (filesystem + database tasks)...")
    horizons = getattr(args, "horizons", None) or config.get("evaluation", {}).get("horizons")
    results = run_controlled_experiment(
        config=config,
        env_root=args.sandbox_dir or "sandbox",
        results_dir=results_dir,
        num_attempts=args.attempts,
        horizons=horizons,
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


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze results and produce all plots. Use --demo for synthetic data."""
    import pandas as pd
    from self_improving_agent.evaluation.metrics import (
        generate_summary_table,
        plot_success_vs_horizon,
        plot_failure_mode_dist,
        plot_cumulative_success,
    )
    from self_improving_agent.analysis.recurrence_analysis import (
        compare_recurrence_across_conditions,
        plot_failure_recurrence,
        print_recurrence_summary,
    )

    results_dir = getattr(args, "output", "results")

    if args.demo:
        from self_improving_agent.experiments.demo_results import generate_demo_results
        print("Generating synthetic demo results...")
        results = generate_demo_results(results_dir=results_dir)
    else:
        csv_path = Path(results_dir) / "controlled_results.csv"
        if not csv_path.exists():
            print(f"No results found at {csv_path}. Run 'python main.py run' first, or use --demo.")
            return
        all_df = pd.read_csv(csv_path)
        results: dict[str, pd.DataFrame] = {
            str(label): grp for label, grp in all_df.groupby("label")
        }

    # Summary table
    summary = generate_summary_table(results)
    print("\n--- Summary Table ---")
    print(summary.to_string(index=False))
    summary.to_csv(Path(results_dir) / "analysis_summary.csv", index=False)

    # Failure recurrence analysis
    print_recurrence_summary(results)
    recur_table = compare_recurrence_across_conditions(results)
    recur_table.to_csv(Path(results_dir) / "recurrence_analysis.csv", index=False)

    # All plots
    plot_success_vs_horizon(results, save_path=str(Path(results_dir) / "success_vs_horizon"))
    plot_failure_mode_dist(results, save_path=str(Path(results_dir) / "failure_mode_dist"))
    plot_cumulative_success(results, save_path=str(Path(results_dir) / "cumulative_success"))
    plot_failure_recurrence(results, save_path=str(Path(results_dir) / "failure_recurrence"))
    print(f"\nAll plots saved to {results_dir}/")


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


def cmd_terminalbench(args: argparse.Namespace) -> None:
    """Run Terminal-Bench with this project's custom agent adapter."""
    import subprocess
    cmd = [sys.executable, "-m", "self_improving_agent.experiments.run_terminal_bench"]
    cmd += ["--dataset", args.dataset]
    for task_id in args.task_id or []:
        cmd += ["--task-id", task_id]
    if args.n_tasks is not None:
        cmd += ["--n-tasks", str(args.n_tasks)]
    cmd += ["--n-concurrent", str(args.n_concurrent)]
    cmd += ["--n-attempts", str(args.n_attempts)]
    cmd += ["--output", args.output]
    if args.run_id:
        cmd += ["--run-id", args.run_id]
    if args.profile:
        cmd += ["--profile", args.profile]
    cmd += ["--max-steps", str(args.max_steps)]
    cmd += ["--command-timeout-sec", str(args.command_timeout_sec)]
    cmd += ["--log-level", args.log_level]
    if args.no_rebuild:
        cmd.append("--no-rebuild")
    if args.no_cleanup:
        cmd.append("--no-cleanup")
    if args.skip_docker_check:
        cmd.append("--skip-docker-check")
    if args.print_command:
        cmd.append("--print-command")
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


def cmd_member2_eval(args: argparse.Namespace) -> None:
    """Run Member 2 prompt evaluation harness."""
    import subprocess
    cmd = [sys.executable, "-m", "self_improving_agent.experiments.member2_eval"]
    if args.mock:
        cmd.append("--mock")
    if args.profile:
        cmd += ["--profile", args.profile]
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    if args.output:
        cmd += ["--output", args.output]
    subprocess.run(cmd, check=True)


def cmd_member2_ablation(args: argparse.Namespace) -> None:
    """Run Member 2 prompt ablation study."""
    import subprocess
    cmd = [sys.executable, "-m", "self_improving_agent.experiments.member2_ablation"]
    if args.mock:
        cmd.append("--mock")
    if args.profile:
        cmd += ["--profile", args.profile]
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    if args.output_dir:
        cmd += ["--output-dir", args.output_dir]
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
    run_p.add_argument("--profile", default=None, help="Model profile: xai | haiku | groq | ollama")

    # --- analyze ---
    ana_p = sub.add_parser("analyze", help="Analyze results and produce all plots")
    ana_p.add_argument("--output", "-o", default="results", help="Directory with results / for output")
    ana_p.add_argument("--demo", action="store_true", help="Use synthetic demo data instead of real results")

    # --- agentbench ---
    ab_p = sub.add_parser("agentbench", help="Run AgentBench OS experiment")
    ab_p.add_argument("--horizons", nargs="+", type=int, default=None)
    ab_p.add_argument("--n-tasks", type=int, default=None)
    ab_p.add_argument("--dry-run", action="store_true")

    # --- terminalbench ---
    tb_p = sub.add_parser("terminalbench", help="Run Terminal-Bench")
    tb_p.add_argument("--dataset", default="terminal-bench-core==0.1.1")
    tb_p.add_argument("--task-id", action="append", default=[])
    tb_p.add_argument("--n-tasks", type=int, default=None)
    tb_p.add_argument("--n-concurrent", type=int, default=1)
    tb_p.add_argument("--n-attempts", type=int, default=1)
    tb_p.add_argument("--output", default="results/terminal_bench")
    tb_p.add_argument("--run-id", default=None)
    tb_p.add_argument("--profile", default=None, help="Model profile: xai | haiku | groq | ollama")
    tb_p.add_argument("--max-steps", type=int, default=50)
    tb_p.add_argument("--command-timeout-sec", type=float, default=180.0)
    tb_p.add_argument("--log-level", default="info")
    tb_p.add_argument("--no-rebuild", action="store_true")
    tb_p.add_argument("--no-cleanup", action="store_true")
    tb_p.add_argument("--skip-docker-check", action="store_true")
    tb_p.add_argument("--print-command", action="store_true")

    # --- ablation ---
    abl_p = sub.add_parser("ablation", help="Run ablation study")
    abl_p.add_argument("--horizon", type=int, default=None)
    abl_p.add_argument("--n-tasks", type=int, default=None)
    abl_p.add_argument("--dry-run", action="store_true")

    # --- member2-eval ---
    m2_p = sub.add_parser("member2-eval", help="Evaluate Member 2 prompts with LLM judge")
    m2_p.add_argument("--dataset", default=None)
    m2_p.add_argument("--output", default=None)
    m2_p.add_argument("--profile", default=None, help="Model profile: xai | haiku | groq | ollama")
    m2_p.add_argument("--mock", action="store_true")

    # --- member2-ablation ---
    m2a_p = sub.add_parser("member2-ablation", help="Run Member 2 prompt ablations")
    m2a_p.add_argument("--dataset", default=None)
    m2a_p.add_argument("--output-dir", default=None)
    m2a_p.add_argument("--profile", default=None, help="Model profile: xai | haiku | groq | ollama")
    m2a_p.add_argument("--mock", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)
    elif args.cmd == "agentbench":
        cmd_agentbench(args)
    elif args.cmd == "terminalbench":
        cmd_terminalbench(args)
    elif args.cmd == "ablation":
        cmd_ablation(args)
    elif args.cmd == "member2-eval":
        cmd_member2_eval(args)
    elif args.cmd == "member2-ablation":
        cmd_member2_ablation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
