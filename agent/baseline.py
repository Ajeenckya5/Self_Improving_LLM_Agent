"""Baseline Plan-Act-Observe agent."""

from tasks.base import Task
from .base import BaseAgent


class BaselineAgent(BaseAgent):
    """Baseline agent without strategy memory."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_type = "baseline"

    def get_system_prompt(self, task: Task, strategies: list[str] | None = None) -> str:
        return """You are an agent that executes long-horizon tasks in a controlled environment.
You must respond with valid JSON only.

Format for each step:
- To take action: {"thought": "brief reasoning", "action": "tool_name", "args": {"param": "value"}}
- To finish: {"thought": "reasoning", "done": true, "summary": "what you accomplished"}

Always list the directory or check the schema first to understand the current state.
Work step by step. Use the correct tools for filesystem or database operations."""
