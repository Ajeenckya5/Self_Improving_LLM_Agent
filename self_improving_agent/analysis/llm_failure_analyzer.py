"""LLM-backed failure analyzer using grounded trace prompts."""

from __future__ import annotations

from typing import Any, Dict

from .categories import FAILURE_CATEGORIES
from .failure_analyzer import FailureAnalyzer
from .prompt_utils import coerce_int_list, load_prompt, parse_json_object, render_prompt, trace_to_text
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LLMFailureAnalyzer:
    """
    Classify task failures with an LLM prompt and fall back to the heuristic
    analyzer if parsing or validation fails.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        llm_client: LLMClient | None = None,
        prompt_name: str | None = None,
        fallback: FailureAnalyzer | None = None,
    ):
        self.config = config
        analysis_cfg = config.get("analysis", {})
        self.prompt_name = prompt_name or analysis_cfg.get(
            "failure_prompt", "failure_analysis.txt"
        )
        self.model = config.get("model", {}).get("analyzer", "grok-4.20-reasoning")
        self.llm = llm_client or LLMClient(config)
        self.fallback = fallback or FailureAnalyzer()
        self._prompt_template = load_prompt(self.prompt_name)

    def analyze(self, task: Dict[str, Any], trace: Any) -> Dict[str, Any]:
        """Return a validated failure analysis JSON object."""
        heuristic_hint = self.fallback.analyze(trace)
        prompt = render_prompt(
            self._prompt_template,
            {
                "task_description": task.get("description", ""),
                "task_id": task.get("id", "unknown"),
                "trace_text": trace_to_text(trace),
                "failure_categories": ", ".join(FAILURE_CATEGORIES + ["other"]),
                "heuristic_failure_type": heuristic_hint.get("failure_type", "other"),
                "heuristic_pattern_summary": heuristic_hint.get("pattern_summary", ""),
            },
        )

        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.0,
            max_tokens=700,
        )

        try:
            parsed = parse_json_object(response)
            result = self._validate(parsed, trace)
            logger.info(
                "LLM failure analysis | prompt=%s | type=%s | steps=%s",
                self.prompt_name,
                result["failure_type"],
                result["failed_steps"],
            )
            return result
        except Exception as exc:
            logger.warning("LLM failure analysis parse failed; using heuristic fallback: %s", exc)
            heuristic_hint["analysis_source"] = "heuristic_fallback"
            return heuristic_hint

    def _validate(self, data: Dict[str, Any], trace: Any) -> Dict[str, Any]:
        failure_type = str(data.get("failure_type", "other")).strip()
        if failure_type not in FAILURE_CATEGORIES and failure_type != "other":
            failure_type = "other"

        failed_steps = coerce_int_list(data.get("failed_steps"))
        pattern_summary = str(data.get("pattern_summary", "")).strip()
        if not pattern_summary:
            pattern_summary = "The LLM analyzer did not provide a failure summary."

        confidence = data.get("confidence", None)
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None

        return {
            "failure_type": failure_type,
            "failed_steps": failed_steps,
            "pattern_summary": pattern_summary,
            "raw_trace_excerpt": trace_to_text(trace, max_steps=10, obs_chars=250),
            "analysis_source": "llm",
            "confidence": confidence,
        }
