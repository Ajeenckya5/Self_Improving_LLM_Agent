"""Base agent with Plan-Act-Observe loop."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from tasks.base import Task
from environment.controlled import ControlledEnvironment
from tracing.logger import TraceLogger, ExecutionTrace


@dataclass
class AgentResult:
    """Result of agent run."""
    success: bool
    steps: int
    trace: ExecutionTrace | None
    message: str


class BaseAgent(ABC):
    """Abstract agent with Plan-Act-Observe loop."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_steps: int = 15,
        openai_api_key: str | None = None,
    ):
        self.model = model
        self.max_steps = max_steps
        self._api_key = openai_api_key
        self.agent_type = "base"

    @abstractmethod
    def get_system_prompt(self, task: Task, strategies: list[str] | None = None) -> str:
        """Build system prompt, optionally with retrieved strategies."""
        pass

    def run(
        self,
        task: Task,
        env: ControlledEnvironment,
        trace_logger: TraceLogger,
        attempt: int = 0,
        strategies: list[str] | None = None,
    ) -> AgentResult:
        """
        Execute Plan-Act-Observe loop until done or max_steps.
        """
        import os
        import json

        api_key = self._api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return AgentResult(
                success=False,
                steps=0,
                trace=None,
                message="OPENAI_API_KEY not set",
            )

        trace_logger.start_trace(task.task_id, self.agent_type, attempt)
        tools = task.get_available_tools()
        tool_descriptions = "\n".join(
            f"- {t['name']}: {t['description']} (params: {t.get('parameters', {})})"
            for t in tools
        )

        system = self.get_system_prompt(task, strategies)
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Task: {task.description}\n\nAvailable tools:\n{tool_descriptions}\n\nStart by listing the directory to see what exists, then plan and execute steps. Respond with JSON: {{\"thought\": \"...\", \"action\": \"tool_name\", \"args\": {{...}}}} or {{\"thought\": \"...\", \"done\": true, \"summary\": \"...\"}}",
            },
        ]

        step = 0
        done = False
        summary = ""

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        except ImportError:
            return AgentResult(
                success=False,
                steps=0,
                trace=None,
                message="openai package not installed",
            )

        while step < self.max_steps and not done:
            step += 1
            state = env.get_state()

            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                )
                content = resp.choices[0].message.content.strip()
            except Exception as e:
                obs = f"LLM error: {e}"
                trace_logger.log_step(state, {"error": str(e)}, obs, "")
                break

            # Parse JSON response
            action_name = None
            args = {}
            reasoning = ""
            try:
                # Extract JSON from response (handle markdown code blocks)
                text = content
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                data = json.loads(text)
                reasoning = data.get("thought", "") or ""
                action_name = data.get("action")
                args = data.get("args", {})
                if data.get("done"):
                    done = True
                    summary = data.get("summary", "")
                    trace_logger.log_step(state, {"done": True, "summary": summary}, "", reasoning)
                    break
            except (json.JSONDecodeError, KeyError):
                # Fallback: try to infer action from text
                obs = f"Could not parse response: {content[:200]}"
                trace_logger.log_step(state, {"raw": content[:100]}, obs, "")
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "Respond with valid JSON only. Use {\"thought\": \"...\", \"action\": \"tool_name\", \"args\": {...}} or {\"done\": true, \"summary\": \"...\"}",
                })
                continue

            if not action_name:
                obs = "No action specified"
                trace_logger.log_step(state, {"content": content[:100]}, obs, reasoning)
                continue

            observation = env.execute(action_name, **args)
            action = {"tool": action_name, "args": args}
            trace_logger.log_step(state, action, observation, reasoning)

            messages.append({"role": "assistant", "content": content})
            done_hint = '{"done": true, "summary": "..."}'
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}\n\nContinue with next action (JSON) or {done_hint} if task is complete.",
            })

        trace = trace_logger.end_trace(False, summary or "Max steps or error")
        return AgentResult(
            success=done,
            steps=step,
            trace=trace,
            message=summary or "Incomplete",
        )
