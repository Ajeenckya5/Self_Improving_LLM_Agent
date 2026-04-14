"""
Generates corrective strategies from failure analysis using an LLM call.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class StrategyGenerator:
    """
    Makes an LLM call to produce a corrective strategy from a failure analysis.
    Reads the prompt template from prompts/strategy_gen.txt.
    """

    def __init__(self, config: Dict[str, Any], llm_client: LLMClient | None = None):
        self.config = config
        self.model = config.get("model", {}).get("analyzer", "gpt-4")
        self.llm = llm_client or LLMClient(config)
        self._prompt_template = self._load_prompt()

    def generate(
        self,
        task: Dict[str, Any],
        failure_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        task : task dict with at least "description"
        failure_analysis : output of FailureAnalyzer.analyze()

        Returns
        -------
        {
            "strategy_text": str,
            "tags": List[str],
            "failure_type": str,
        }
        """
        failed_steps_text = (
            ", ".join(str(s) for s in failure_analysis.get("failed_steps", []))
            or "unknown"
        )

        # Use manual replacement instead of .format() to avoid conflicts
        # with literal {braces} in the JSON example inside the prompt template.
        prompt = self._prompt_template
        for key, value in [
            ("{task_description}", task.get("description", "")),
            ("{failure_type}", failure_analysis.get("failure_type", "other")),
            ("{pattern_summary}", failure_analysis.get("pattern_summary", "")),
            ("{failed_steps_text}", failed_steps_text),
        ]:
            prompt = prompt.replace(key, value)

        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.4,
            max_tokens=500,
        )

        parsed = self._parse_response(response, failure_analysis)
        logger.info(
            "Strategy generated | failure_type=%s | tags=%s",
            parsed["failure_type"],
            parsed["tags"],
        )
        return parsed

    # ------------------------------------------------------------------

    def _parse_response(
        self,
        response: str,
        failure_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Try JSON parse first
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return {
                    "strategy_text": data.get("strategy_text", response.strip()),
                    "tags": data.get("tags", []),
                    "failure_type": failure_analysis.get("failure_type", "other"),
                }
            except json.JSONDecodeError:
                pass

        # Fallback: treat entire response as strategy text
        logger.warning("Could not parse JSON from strategy generator response; using raw text.")
        return {
            "strategy_text": response.strip()[:500],
            "tags": [failure_analysis.get("failure_type", "other")],
            "failure_type": failure_analysis.get("failure_type", "other"),
        }

    def _load_prompt(self) -> str:
        path = _PROMPTS_DIR / "strategy_gen.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        # Inline fallback
        return (
            "An LLM agent failed a task. Generate a corrective strategy.\n\n"
            "Task: {task_description}\n"
            "Failure type: {failure_type}\n"
            "Failure summary: {pattern_summary}\n"
            "Failed steps: {failed_steps_text}\n\n"
            "Write a corrective strategy (2-4 sentences) that tells a future agent:\n"
            "1. What went wrong in this type of task\n"
            "2. Specifically what to do differently\n\n"
            "Also provide 3-5 keyword tags for retrieval.\n\n"
            'Respond in JSON:\n{{"strategy_text": "...", "tags": ["...", "..."]}}'
        )
