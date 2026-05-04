"""Terminal-Bench adapter for the self-improving agent framework."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

from ..utils.llm_client import LLMClient


ACTION_PATTERN = re.compile(
    r"Action:\s*(?P<tool>bash|finish)\((?P<args>.*)\)\s*$",
    re.IGNORECASE | re.DOTALL,
)


class SelfImprovingTerminalBenchAgent(BaseAgent):
    """Run this project's LLM loop through Terminal-Bench's tmux session API."""

    @staticmethod
    def name() -> str:
        return "self-improving-agent"

    def __init__(
        self,
        config_path: str | None = None,
        profile: str | None = None,
        max_steps: int = 50,
        command_timeout_sec: float = 180.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._package_root = Path(__file__).resolve().parents[1]
        self._repo_root = Path(__file__).resolve().parents[2]
        self._config_path = (
            Path(config_path) if config_path else self._package_root / "config.yaml"
        )
        self._config = self._load_config(profile)
        self._llm = LLMClient(self._config)
        self._max_steps = int(max_steps)
        self._command_timeout_sec = float(command_timeout_sec)
        self._model = self._config.get("model", {}).get("primary", "gpt-4")
        self._temperature = self._config.get("agent", {}).get("temperature", 0.7)
        self._max_tokens = self._config.get("agent", {}).get("max_tokens", 1000)
        self._prompt = (
            self._package_root / "prompts" / "terminal_bench_agent.txt"
        ).read_text(encoding="utf-8")

    def _load_config(self, profile: str | None) -> dict[str, Any]:
        with self._config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)

        active = profile or config.get("active_profile")
        profiles = config.get("model_profiles", {})
        if active in profiles:
            config.setdefault("model", {}).update(profiles[active])
        return config

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        instruction = self._render_instruction(instruction)
        messages = [
            {"role": "system", "content": self._prompt},
            {
                "role": "user",
                "content": (
                    f"Task:\n{instruction}\n\n"
                    f"Initial terminal state:\n{session.capture_pane()}"
                ),
            },
        ]
        markers: list[tuple[float, str]] = []

        if logging_dir is not None:
            logging_dir.mkdir(parents=True, exist_ok=True)
            (logging_dir / "instruction.txt").write_text(instruction, encoding="utf-8")

        for step in range(1, self._max_steps + 1):
            response = self._llm.chat(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            markers.append((session.get_asciinema_timestamp(), response))

            if logging_dir is not None:
                (logging_dir / f"step_{step:03d}_response.txt").write_text(
                    response,
                    encoding="utf-8",
                )

            action = self._parse_action(response)
            if action is None:
                return AgentResult(
                    failure_mode=FailureMode.PARSE_ERROR,
                    timestamped_markers=markers,
                )

            tool, args = action
            if tool == "finish":
                return AgentResult(
                    failure_mode=FailureMode.NONE,
                    timestamped_markers=markers,
                )

            try:
                session.send_keys(
                    [args, "Enter"],
                    block=True,
                    max_timeout_sec=self._command_timeout_sec,
                )
                observation = session.get_incremental_output()
            except TimeoutError as exc:
                observation = (
                    f"Command timed out after {self._command_timeout_sec} seconds: {exc}\n"
                    f"{session.capture_pane()}"
                )

            if logging_dir is not None:
                (logging_dir / f"step_{step:03d}_observation.txt").write_text(
                    observation,
                    encoding="utf-8",
                )

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})

        return AgentResult(
            failure_mode=FailureMode.UNKNOWN_AGENT_ERROR,
            timestamped_markers=markers,
        )

    def _parse_action(self, response: str) -> tuple[str, str] | None:
        match = ACTION_PATTERN.search(response.strip())
        if not match:
            return None

        tool = match.group("tool").lower()
        args = match.group("args").strip()
        if tool == "bash":
            args = args.strip().strip('"').strip("'")
            if not args:
                return None
        return tool, args


def write_terminal_bench_agent_info(path: str | Path) -> Path:
    """Write metadata useful for debugging Terminal-Bench custom-agent runs."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "agent_import_path": (
                    "self_improving_agent.integrations.terminal_bench_agent:"
                    "SelfImprovingTerminalBenchAgent"
                ),
                "agent_name": SelfImprovingTerminalBenchAgent.name(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path
