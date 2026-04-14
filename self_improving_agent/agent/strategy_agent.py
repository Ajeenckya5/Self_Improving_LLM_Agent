"""
Strategy-guided agent: retrieves relevant past strategies before running ReAct.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base_agent import AgentTrace, BaseAgent
from ..utils.logger import get_logger

logger = get_logger(__name__)

STRATEGY_INJECT_TEMPLATE = """You are an LLM agent about to attempt a task.
Based on past experience, here are relevant strategies to keep in mind:

{strategies_text}

Apply these strategies proactively. Now, attempt the following task:

Task: {task_description}
Available tools: bash(cmd), read_file(path), write_file(path, content), check_output(cmd), navigate(url), click(selector), type(selector, text), finish(result)
"""


class StrategyAgent(BaseAgent):
    """
    Our method: strategy-guided ReAct agent.
    Retrieves relevant strategies from memory and injects them into the system prompt.
    """

    AGENT_TYPE = "strategy"

    def __init__(self, config: Dict[str, Any], llm_client=None, retriever=None):
        super().__init__(config, llm_client)
        self.retriever = retriever  # memory.retriever.Retriever instance

    def run(
        self,
        task: Dict[str, Any],
        strategies: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, AgentTrace]:
        # If retriever is available and no strategies were pre-supplied, fetch them
        if strategies is None and self.retriever is not None:
            task_description = task.get("description", "")
            try:
                query_embedding = self.retriever.embed(task_description)
                top_k = self.config.get("memory", {}).get("top_k", 3)
                threshold = self.config.get("memory", {}).get("similarity_threshold", 0.65)
                strategies = self.retriever.retrieve(query_embedding, top_k=top_k, threshold=threshold)
                logger.info(
                    "Retrieved %d strategies for task %s",
                    len(strategies),
                    task.get("id", "unknown"),
                )
            except Exception as exc:
                logger.warning("Strategy retrieval failed: %s", exc)
                strategies = []

        return super().run(task, strategies or [])

    # ------------------------------------------------------------------

    def _build_system_prompt(self, strategies: List[Dict[str, Any]]) -> str:
        if not strategies:
            return super()._build_system_prompt(strategies)

        strategies_text = "\n".join(
            f"{i+1}. {s.get('strategy_text', '').strip()}"
            for i, s in enumerate(strategies)
        )

        tools_desc = (
            "Available tools: bash(cmd), read_file(path), write_file(path, content), "
            "check_output(cmd), navigate(url), click(selector), type(selector, text), finish(result)"
        )

        return (
            "You are a capable AI agent solving tasks step by step.\n\n"
            "Format EVERY response as:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool_name>(<arguments>)\n\n"
            f"{tools_desc}\n\n"
            "Based on past experience with similar tasks, keep these strategies in mind:\n"
            f"{strategies_text}\n\n"
            "Apply these strategies proactively. Think carefully before acting. "
            "Avoid repeating the same action if it already failed."
        )
