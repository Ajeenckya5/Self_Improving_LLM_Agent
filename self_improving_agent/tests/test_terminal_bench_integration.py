"""Tests for the optional Terminal-Bench integration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from terminal_bench.agents.failure_mode import FailureMode

from ..experiments.run_terminal_bench import AGENT_IMPORT_PATH, build_tb_command
from ..integrations.harbor_agent import SelfImprovingHarborAgent
from ..integrations.terminal_bench_agent import SelfImprovingTerminalBenchAgent


class FakeLLMClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.messages: list[list[dict[str, str]]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        self.messages.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return "Thought: done\nAction: finish(done)"


class FakeTmuxSession:
    def __init__(self):
        self.commands: list[list[str]] = []

    def capture_pane(self, capture_entire: bool = False) -> str:
        return "fake terminal screen"

    def send_keys(
        self,
        keys: list[str],
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ) -> None:
        self.commands.append(keys)

    def get_incremental_output(self) -> str:
        return "New Terminal Output:\ncommand completed"

    def get_asciinema_timestamp(self) -> float:
        return 0.0


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "active_profile": "mock",
        "model_profiles": {},
        "model": {
            "primary": "gpt-4",
            "analyzer": "gpt-4",
            "embedding": "all-MiniLM-L6-v2",
        },
        "agent": {"max_steps": 4, "temperature": 0.0, "max_tokens": 300},
        "logging": {"llm_calls_log": str(tmp_path / "llm_calls.jsonl")},
    }


def test_build_terminal_bench_command_includes_custom_agent() -> None:
    args = argparse.Namespace(
        dataset="terminal-bench-core==0.1.1",
        output="results/terminal_bench",
        n_concurrent=1,
        n_attempts=1,
        config="self_improving_agent/config.yaml",
        max_steps=25,
        command_timeout_sec=120.0,
        profile="xai",
        run_id="smoke",
        n_tasks=1,
        task_id=["hello-world"],
        no_rebuild=True,
        no_cleanup=False,
        log_level="warning",
    )

    cmd = build_tb_command(args)

    assert cmd[:2] == ["tb", "run"]
    assert ["--agent-import-path", AGENT_IMPORT_PATH] == cmd[4:6]
    assert "--task-id" in cmd
    assert "hello-world" in cmd
    assert "--n-tasks" in cmd
    assert "profile=xai" in cmd
    assert "--no-rebuild" in cmd


def test_terminal_bench_agent_executes_bash_action(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(_config(tmp_path)), encoding="utf-8")
    monkeypatch.setenv("MOCK_LLM", "1")

    agent = SelfImprovingTerminalBenchAgent(
        config_path=str(config_path),
        max_steps=3,
        command_timeout_sec=5,
    )
    agent._llm = FakeLLMClient(
        [
            "Thought: create the marker file\nAction: bash(touch done.txt)",
            "Thought: verified\nAction: finish(done)",
        ]
    )
    session = FakeTmuxSession()

    result = agent.perform_task("Create done.txt", session)  # type: ignore[arg-type]

    assert result.failure_mode == FailureMode.NONE
    assert session.commands == [["touch done.txt", "Enter"]]


def test_harbor_agent_executes_bash_action(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from harbor.environments.base import ExecResult
    from harbor.models.agent.context import AgentContext
    import yaml

    class FakeHarborEnvironment:
        def __init__(self):
            self.commands: list[str] = []

        async def exec(
            self,
            command: str,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_sec: int | None = None,
            user: str | int | None = None,
        ) -> ExecResult:
            self.commands.append(command)
            return ExecResult(stdout="command completed", stderr="", return_code=0)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(tmp_path)), encoding="utf-8")
    monkeypatch.setenv("MOCK_LLM", "1")

    agent = SelfImprovingHarborAgent(
        logs_dir=tmp_path / "logs",
        config_path=str(config_path),
        max_steps=3,
        command_timeout_sec=5,
    )
    agent._llm = FakeLLMClient(
        [
            "Thought: create the marker file\nAction: bash(touch done.txt)",
            "Thought: verified\nAction: finish(done)",
        ]
    )
    env = FakeHarborEnvironment()
    context = AgentContext()

    import asyncio

    asyncio.run(agent.run("Create done.txt", env, context))  # type: ignore[arg-type]

    assert env.commands == ["touch done.txt"]
    assert context.metadata is not None
    assert context.metadata["completed"] is True
