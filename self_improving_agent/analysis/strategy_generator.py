"""
Generates corrective strategies from failure analysis using an LLM call.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .prompt_utils import load_prompt, parse_json_object, render_prompt
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StrategyGenerator:
    """
    Makes an LLM call to produce a corrective strategy from a failure analysis.
    Reads the prompt template from prompts/strategy_gen.txt.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        llm_client: LLMClient | None = None,
        prompt_name: str | None = None,
    ):
        self.config = config
        self.model = config.get("model", {}).get("analyzer", "gpt-4")
        self.prompt_name = prompt_name or config.get("analysis", {}).get(
            "strategy_prompt", "strategy_gen.txt"
        )
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

        prompt = render_prompt(
            self._prompt_template,
            {
                "task_description": task.get("description", ""),
                "task_id": task.get("id", "unknown"),
                "failure_type": failure_analysis.get("failure_type", "other"),
                "pattern_summary": failure_analysis.get("pattern_summary", ""),
                "failed_steps_text": failed_steps_text,
                "raw_trace_excerpt": failure_analysis.get("raw_trace_excerpt", ""),
            },
        )

        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.2,
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
        try:
            data = parse_json_object(response)
            strategy_text = str(data.get("strategy_text", "")).strip()
            if not strategy_text:
                strategy_text = response.strip()

            tags = data.get("tags", [])
            if not isinstance(tags, list):
                tags = re.split(r"[,;\n]+", str(tags))
            tags = [str(tag).strip().lower().replace(" ", "_") for tag in tags if str(tag).strip()]

            decision_rule = str(data.get("decision_rule", "")).strip()
            if decision_rule and decision_rule not in strategy_text:
                strategy_text = f"{strategy_text} Decision rule: {decision_rule}"

            return {
                "strategy_text": strategy_text[:1200],
                "tags": tags[:8],
                "failure_type": failure_analysis.get("failure_type", "other"),
            }
        except Exception:
            pass

        # Fallback: treat entire response as strategy text
        logger.warning("Could not parse JSON from strategy generator response; using raw text.")
        return {
            "strategy_text": response.strip()[:500],
            "tags": [failure_analysis.get("failure_type", "other")],
            "failure_type": failure_analysis.get("failure_type", "other"),
        }

    def _load_prompt(self) -> str:
        return load_prompt(self.prompt_name)
