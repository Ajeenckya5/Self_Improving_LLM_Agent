"""Run Terminal-Bench with this project's custom agent adapter."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


AGENT_IMPORT_PATH = (
    "self_improving_agent.integrations.terminal_bench_agent:"
    "SelfImprovingTerminalBenchAgent"
)
DEFAULT_DATASET = "terminal-bench-core==0.1.1"


def build_tb_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        "tb",
        "run",
        "--dataset",
        args.dataset,
        "--agent-import-path",
        AGENT_IMPORT_PATH,
        "--output-path",
        args.output,
        "--n-concurrent",
        str(args.n_concurrent),
        "--n-attempts",
        str(args.n_attempts),
        "--agent-kwarg",
        f"config_path={args.config}",
        "--agent-kwarg",
        f"max_steps={args.max_steps}",
        "--agent-kwarg",
        f"command_timeout_sec={args.command_timeout_sec}",
    ]

    if args.profile:
        cmd.extend(["--agent-kwarg", f"profile={args.profile}"])
    if args.run_id:
        cmd.extend(["--run-id", args.run_id])
    if args.n_tasks is not None:
        cmd.extend(["--n-tasks", str(args.n_tasks)])
    for task_id in args.task_id or []:
        cmd.extend(["--task-id", task_id])
    if args.no_rebuild:
        cmd.append("--no-rebuild")
    if args.no_cleanup:
        cmd.append("--no-cleanup")
    if args.log_level:
        cmd.extend(["--log-level", args.log_level])
    return cmd


def terminal_bench_ready(skip_docker_check: bool = False) -> tuple[bool, str]:
    if shutil.which("tb") is None:
        return False, "Terminal-Bench CLI `tb` is not installed."
    if skip_docker_check:
        return True, "Terminal-Bench CLI found; Docker check skipped."

    docker_path = shutil.which("docker")
    if docker_path is None:
        return False, "Docker CLI is not installed."

    result = subprocess.run(
        [docker_path, "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        return False, (
            "Docker CLI is installed, but the Docker daemon is not reachable. "
            "Start Docker Desktop, then rerun this command."
        )
    return True, f"Docker daemon reachable: {result.stdout.strip()}"


def main(args: argparse.Namespace) -> None:
    cmd = build_tb_command(args)
    if args.print_command:
        print(" ".join(shlex_quote(part) for part in cmd))
        return

    ready, message = terminal_bench_ready(args.skip_docker_check)
    if not ready:
        raise SystemExit(message)
    print(message)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Terminal-Bench with this agent")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--output", default="results/terminal_bench")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--command-timeout-sec", type=float, default=180.0)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--no-rebuild", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--skip-docker-check", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


if __name__ == "__main__":
    main(parse_args())
