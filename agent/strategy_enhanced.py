"""Strategy-enhanced agent that uses retrieved strategies."""

from tasks.base import Task
from .base import BaseAgent


class StrategyEnhancedAgent(BaseAgent):
    """Agent that injects retrieved strategies into the prompt."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_type = "strategy_enhanced"

    def get_system_prompt(self, task: Task, strategies: list[str] | None = None) -> str:
        base = """You are an agent that executes long-horizon tasks in a controlled environment.
You must respond with valid JSON only.

Format for each step:
- To take action: {"thought": "brief reasoning", "action": "tool_name", "args": {"param": "value"}}
- To finish: {"thought": "reasoning", "done": true, "summary": "what you accomplished"}

Always list the directory or check the schema first to understand the current state.
Work step by step. Use the correct tools for filesystem or database operations."""

        if strategies:
            base += "\n\n**Important strategies from similar past failures:**\n"
            for i, s in enumerate(strategies, 1):
                base += f"{i}. {s}\n"
            base += "\nApply these strategies to avoid repeating past mistakes."
        return base
