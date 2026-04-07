"""
Agent evaluation framework: baseline vs strategy-enhanced agent.

Run experiments:
    python main.py run

Set OPENAI_API_KEY in .env or environment.
"""

import argparse
import os
from pathlib import Path

# Load .env if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import json
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from config import ENV_ROOT, TRACES_DIR, STRATEGY_DB_PATH, CHROMA_PATH, MAX_STEPS, EXPERIMENT_SEED
from experiments.runner import ExperimentRunner
from experiments.metrics import ExperimentMetrics, produce_plots
from strategy_memory.store import StrategyMemory


def main():
    parser = argparse.ArgumentParser(description="Agent evaluation framework")
    sub = parser.add_subparsers(dest="cmd", help="Command")

    run_p = sub.add_parser("run", help="Run experiment")
    run_p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    run_p.add_argument("--attempts", type=int, default=2, help="Attempts per task per agent")
    run_p.add_argument("--seed", type=int, default=EXPERIMENT_SEED)
    run_p.add_argument("--output", "-o", default="results", help="Output dir for plots")
    run_p.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    run_p.add_argument("--sandbox-dir", help="Override sandbox dir (use /tmp/agent_sandbox if permission errors)")

    args = parser.parse_args()

    if args.cmd == "run":
        project_root = Path(__file__).parent
        env_root = Path(getattr(args, "sandbox_dir", None) or project_root / "sandbox")
        traces_dir = project_root / "traces"
        env_root.mkdir(parents=True, exist_ok=True)
        traces_dir.mkdir(parents=True, exist_ok=True)

        strategy_mem = StrategyMemory(persist_path=project_root / "chroma_db")

        runner = ExperimentRunner(
            env_root=env_root,
            traces_dir=traces_dir,
            strategy_memory=strategy_mem,
            max_steps=args.max_steps,
            num_attempts=args.attempts,
            seed=args.seed,
        )

        print("Running experiment (baseline then strategy-enhanced)...")
        results = runner.run_experiment()

        print("\n--- Results ---")
        print(json.dumps(results, indent=2))

        if not args.no_plot:
            out_dir = Path(args.output)
            paths = produce_plots(runner.metrics, str(out_dir))
            print(f"\nPlots saved to: {paths}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
