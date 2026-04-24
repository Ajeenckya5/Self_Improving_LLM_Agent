"""LLM-as-judge harness for Member 2 analyzer and strategy outputs."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..analysis.prompt_utils import coerce_int_list, load_prompt, parse_json_object, render_prompt, trace_to_text
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FailureStrategyJudge:
    """
    Scores whether a failure analysis and generated strategy are correct,
    grounded, actionable, and useful for retrieval.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        llm_client: LLMClient | None = None,
        prompt_name: str | None = None,
    ):
        self.config = config
        analysis_cfg = config.get("analysis", {})
        self.prompt_name = prompt_name or analysis_cfg.get("judge_prompt", "evaluation_judge.txt")
        self.model = analysis_cfg.get(
            "judge_model",
            config.get("model", {}).get("judge", config.get("model", {}).get("analyzer", "gpt-4")),
        )
        self.llm = llm_client or LLMClient(config)
        self._prompt_template = load_prompt(self.prompt_name)

    def evaluate(
        self,
        task: Dict[str, Any],
        trace: Any,
        failure_analysis: Dict[str, Any],
        strategy: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return judge metrics as a plain dict."""
        prompt = render_prompt(
            self._prompt_template,
            {
                "task_description": task.get("description", ""),
                "gold_failure_type": task.get("gold_failure_type", task.get("failure_type", "")),
                "gold_failed_steps": task.get("gold_failed_steps", task.get("failed_steps", [])),
                "trace_text": trace_to_text(trace, max_steps=30),
                "candidate_failure_analysis": json.dumps(failure_analysis, indent=2),
                "candidate_strategy": json.dumps(strategy, indent=2),
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
        except Exception as exc:
            logger.warning("Judge response parse failed; using deterministic fallback: %s", exc)
            parsed = {}

        return self._normalize(parsed, task, failure_analysis, strategy)

    def _normalize(
        self,
        parsed: Dict[str, Any],
        task: Dict[str, Any],
        failure_analysis: Dict[str, Any],
        strategy: Dict[str, Any],
    ) -> Dict[str, Any]:
        gold_type = task.get("gold_failure_type", task.get("failure_type"))
        pred_type = failure_analysis.get("failure_type")
        type_correct = bool(parsed.get("failure_type_correct", pred_type == gold_type))

        gold_steps = coerce_int_list(task.get("gold_failed_steps", task.get("failed_steps")))
        pred_steps = coerce_int_list(failure_analysis.get("failed_steps"))
        overlap = parsed.get("failed_steps_overlap", _jaccard(gold_steps, pred_steps))

        result = {
            "failure_type_correct": type_correct,
            "failed_steps_overlap": _coerce_float(overlap, 0.0),
            "analysis_grounding_score": _coerce_int(parsed.get("analysis_grounding_score"), 1),
            "strategy_specificity_score": _coerce_int(parsed.get("strategy_specificity_score"), 1),
            "strategy_actionability_score": _coerce_int(parsed.get("strategy_actionability_score"), 1),
            "retrieval_tags_score": _coerce_int(parsed.get("retrieval_tags_score"), 1),
            "overall_score": _coerce_int(parsed.get("overall_score"), 1),
            "rationale": str(parsed.get("rationale", "")).strip(),
            "strategy_length": len(str(strategy.get("strategy_text", "")).split()),
            "tag_count": len(strategy.get("tags", []) or []),
        }
        return result


def _jaccard(left: list[int], right: list[int]) -> float:
    if not left and not right:
        return 1.0
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
