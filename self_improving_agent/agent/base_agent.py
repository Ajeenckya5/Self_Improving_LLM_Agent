"""
ReAct-style base agent that runs a Thought → Action → Observation loop.
Records every step as a structured trace and returns (success, trace).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger(__name__)

TOOLS = ["bash", "read_file", "write_file", "check_output", "navigate", "click", "type"]

TOOL_PATTERN = re.compile(
    r"Action:\s*(bash|read_file|write_file|check_output|navigate|click|type)\((.+?)\)\s*$",
    re.MULTILINE | re.DOTALL,
)

FINISH_PATTERN = re.compile(r"Action:\s*finish\((.*?)\)", re.IGNORECASE | re.DOTALL)


@dataclass
class TraceStep:
    step: int
    thought: str
    action: str
    observation: str
    success: bool = False


@dataclass
class AgentTrace:
    task_id: str
    task_description: str
    steps: List[TraceStep] = field(default_factory=list)
    final_success: bool = False
    total_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "steps": [
                {
                    "step": s.step,
                    "thought": s.thought,
                    "action": s.action,
                    "observation": s.observation,
                    "success": s.success,
                }
                for s in self.steps
            ],
            "final_success": self.final_success,
            "total_steps": self.total_steps,
        }

    def to_text(self, max_steps: int = 20) -> str:
        lines = []
        for s in self.steps[:max_steps]:
            lines.append(f"Step {s.step}:")
            lines.append(f"  Thought: {s.thought}")
            lines.append(f"  Action: {s.action}")
            obs = s.observation[:300] + "..." if len(s.observation) > 300 else s.observation
            lines.append(f"  Observation: {obs}")
        return "\n".join(lines)


class BaseAgent:
    """
    ReAct-style agent. Subclass and override `_build_system_prompt` or
    `_build_user_prompt` to customise behaviour.
    """

    AGENT_TYPE = "react"

    def __init__(self, config: Dict[str, Any], llm_client: Optional[LLMClient] = None):
        self.config = config
        self.max_steps: int = config.get("agent", {}).get("max_steps", 25)
        self.temperature: float = config.get("agent", {}).get("temperature", 0.7)
        self.max_tokens: int = config.get("agent", {}).get("max_tokens", 1000)
        self.model: str = config.get("model", {}).get("primary", "gpt-4")
        self.llm = llm_client or LLMClient(config)
        self._sandbox_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        task: Dict[str, Any],
        strategies: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, AgentTrace]:
        """Execute a task and return (success, trace)."""
        task_id = task.get("id", "unknown")
        task_description = task.get("description", "")
        env = task.get("env")  # environment object if provided

        trace = AgentTrace(
            task_id=task_id,
            task_description=task_description,
        )

        messages = self._build_initial_messages(task, strategies or [])
        context_limit = 6000  # rough token budget for observations

        for step_num in range(1, self.max_steps + 1):
            # Ask the LLM for the next thought + action
            response_text = self.llm.chat(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            thought, action_str = self._parse_response(response_text)

            # Check for explicit finish
            finish_match = FINISH_PATTERN.search(action_str)
            if finish_match or action_str.strip().lower().startswith("finish"):
                success_flag = self._check_task_success(task, env, action_str)
                trace.steps.append(
                    TraceStep(
                        step=step_num,
                        thought=thought,
                        action=action_str,
                        observation="Task declared complete.",
                        success=success_flag,
                    )
                )
                trace.final_success = success_flag
                trace.total_steps = step_num
                return success_flag, trace

            # Execute the action
            observation = self._execute_action(action_str, env)

            # Guard against context explosion
            if len(observation) > context_limit:
                observation = observation[:context_limit] + "\n[TRUNCATED: observation exceeded context limit]"

            step_success = not observation.lower().startswith("error")
            step = TraceStep(
                step=step_num,
                thought=thought,
                action=action_str,
                observation=observation,
                success=step_success,
            )
            trace.steps.append(step)

            # Append to messages for next iteration
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

            logger.debug("Step %d | action=%s | obs_len=%d", step_num, action_str[:60], len(observation))

        # Ran out of steps
        trace.final_success = False
        trace.total_steps = self.max_steps
        return False, trace

    # ------------------------------------------------------------------
    # Prompt construction (override in subclasses)
    # ------------------------------------------------------------------

    def _build_initial_messages(
        self,
        task: Dict[str, Any],
        strategies: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        system_prompt = self._build_system_prompt(strategies)
        user_prompt = self._build_user_prompt(task)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_system_prompt(self, strategies: List[Dict[str, Any]]) -> str:
        tools_desc = (
            "Available tools:\n"
            "  bash(cmd)                    - run a shell command\n"
            "  read_file(path)              - read a file's content\n"
            "  write_file(path, content)    - write content to a file\n"
            "  check_output(cmd)            - run cmd and return stdout\n"
            "  navigate(url)                - open a URL in the browser\n"
            "  click(selector)              - click a CSS/XPath selector\n"
            "  type(selector, text)         - type text into a field\n"
            "  finish(result)               - declare the task complete\n"
        )
        return (
            "You are a capable AI agent solving tasks step by step.\n\n"
            "Format EVERY response as:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool_name>(<arguments>)\n\n"
            f"{tools_desc}\n"
            "Think carefully before acting. Avoid repeating the same action if it already failed."
        )

    def _build_user_prompt(self, task: Dict[str, Any]) -> str:
        desc = task.get("description", "")
        return f"Task: {desc}\n\nBegin."

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(self, action_str: str, env: Any) -> str:
        # Try environment first (it may handle all tools)
        if env is not None and hasattr(env, "step"):
            return env.step(action_str)

        # Local fallback execution
        match = TOOL_PATTERN.search(action_str)
        if not match:
            return "Error: Could not parse action. Use the format: Action: tool_name(arguments)"

        tool_name = match.group(1).strip()
        raw_args = match.group(2).strip()

        try:
            if tool_name == "bash":
                return self._tool_bash(raw_args)
            elif tool_name == "read_file":
                return self._tool_read_file(raw_args)
            elif tool_name == "write_file":
                return self._tool_write_file(raw_args)
            elif tool_name == "check_output":
                return self._tool_bash(raw_args)
            else:
                return f"Error: Tool '{tool_name}' requires an environment to be provided."
        except Exception as exc:
            return f"Error: {exc}"

    # ------------------------------------------------------------------
    # Built-in tool implementations
    # ------------------------------------------------------------------

    def _tool_bash(self, cmd: str) -> str:
        cmd = cmd.strip().strip('"').strip("'")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            out = result.stdout + result.stderr
            return out[:2000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds."
        except Exception as exc:
            return f"Error: {exc}"

    def _tool_read_file(self, path: str) -> str:
        path = path.strip().strip('"').strip("'")
        try:
            content = Path(path).read_text(encoding="utf-8", errors="replace")
            return content[:2000]
        except FileNotFoundError:
            return f"Error: File not found: {path}"
        except Exception as exc:
            return f"Error: {exc}"

    def _tool_write_file(self, args: str) -> str:
        # Expect: path, content  (first comma-separated token is path)
        parts = args.split(",", 1)
        if len(parts) < 2:
            return "Error: write_file requires (path, content)"
        path = parts[0].strip().strip('"').strip("'")
        content = parts[1].strip().strip('"').strip("'")
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Written {len(content)} bytes to {path}"
        except Exception as exc:
            return f"Error: {exc}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_response(self, text: str) -> Tuple[str, str]:
        thought = ""
        action = ""
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.+?)$", text, re.DOTALL | re.MULTILINE)
        if thought_match:
            thought = thought_match.group(1).strip()
        if action_match:
            action = "Action: " + action_match.group(1).strip()
        if not thought and not action:
            thought = text
            action = "Action: finish(no parseable action)"
        return thought, action

    def _check_task_success(self, task: Dict[str, Any], env: Any, action_str: str) -> bool:
        if env is not None and hasattr(env, "is_success"):
            return env.is_success()
        # Default: treat finish() as success unless explicitly failed
        return "fail" not in action_str.lower()
