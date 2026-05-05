"""Harbor external-agent adapter for Terminal-Bench 2.0 runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from ..utils.llm_client import LLMClient


ACTION_PATTERN = re.compile(
    r"Action:\s*(?P<tool>bash|finish)\((?P<args>.*)\)\s*$",
    re.IGNORECASE | re.DOTALL,
)


class SelfImprovingHarborAgent(BaseAgent):
    """External Harbor agent using this repo's LLM client and ReAct prompt."""

    SUPPORTS_ATIF = False
    SUPPORTS_WINDOWS = False

    @staticmethod
    def name() -> str:
        return "self-improving-agent"

    def version(self) -> str:
        return "0.1.0"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        config_path: str | None = None,
        profile: str | None = None,
        max_steps: int = 50,
        command_timeout_sec: int = 180,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._package_root = Path(__file__).resolve().parents[1]
        self._config_path = (
            Path(config_path) if config_path else self._package_root / "config.yaml"
        )
        self._config = self._load_config(profile)
        if model_name:
            self._config.setdefault("model", {})["primary"] = model_name
        self._llm = LLMClient(self._config)
        self._max_steps = int(max_steps)
        self._command_timeout_sec = int(command_timeout_sec)
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

    async def setup(self, environment: BaseEnvironment) -> None:
        """No in-container install is needed for the external adapter."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "instruction.txt").write_text(instruction, encoding="utf-8")

        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": f"Task:\n{instruction}\n\nBegin."},
        ]
        steps: list[dict[str, Any]] = []

        for step in range(1, self._max_steps + 1):
            response = self._llm.chat(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            (self.logs_dir / f"step_{step:03d}_response.txt").write_text(
                response,
                encoding="utf-8",
            )

            parsed = self._parse_action(response)
            if parsed is None:
                steps.append(
                    {
                        "step": step,
                        "response": response,
                        "error": "Could not parse action.",
                    }
                )
                break

            tool, args = parsed
            if tool == "finish":
                steps.append({"step": step, "action": "finish", "summary": args})
                break

            result = await environment.exec(
                command=args,
                timeout_sec=self._command_timeout_sec,
            )
            observation = self._format_observation(result.return_code, result.stdout, result.stderr)
            (self.logs_dir / f"step_{step:03d}_observation.txt").write_text(
                observation,
                encoding="utf-8",
            )

            steps.append(
                {
                    "step": step,
                    "action": args,
                    "return_code": result.return_code,
                    "stdout_preview": (result.stdout or "")[:1000],
                    "stderr_preview": (result.stderr or "")[:1000],
                }
            )
            context.metadata = {
                "agent": self.name(),
                "steps": steps,
                "completed": False,
            }
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})

        context.metadata = {
            "agent": self.name(),
            "steps": steps,
            "completed": bool(steps and steps[-1].get("action") == "finish"),
        }

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

    def _format_observation(
        self,
        return_code: int,
        stdout: str | None,
        stderr: str | None,
    ) -> str:
        parts = [f"return_code={return_code}"]
        if stdout:
            parts.append(f"stdout:\n{stdout[:4000]}")
        if stderr:
            parts.append(f"stderr:\n{stderr[:4000]}")
        return "\n\n".join(parts)
