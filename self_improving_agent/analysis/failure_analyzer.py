"""
Analyzes execution traces to identify failure patterns without LLM calls.
Produces a structured failure report for use by the strategy generator.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)

CONTEXT_TOKEN_THRESHOLD = 0.8  # 80 % of assumed context window
ASSUMED_CONTEXT_TOKENS = 4096
MAX_OBS_TOKENS = int(ASSUMED_CONTEXT_TOKENS * CONTEXT_TOKEN_THRESHOLD)

# Rough chars-per-token approximation
CHARS_PER_TOKEN = 4


class FailureAnalyzer:
    """
    Identifies failure patterns in a failed execution trace.

    Failure types detected:
      - repeated_action:    same action string ≥ 3 times with no progress
      - circular_loop:      action sequence that repeats as a subsequence
      - context_truncation: any observation contains a truncation marker or is very long
      - incorrect_reasoning: agent contradicts a prior successful observation
      - tool_misuse:        action cannot be parsed / tool returns unexpected type
    """

    def analyze(self, trace: Any) -> Dict[str, Any]:
        """
        Parameters
        ----------
        trace : AgentTrace or dict  (both work)

        Returns
        -------
        {
            "failure_type": str,
            "failed_steps": List[int],
            "pattern_summary": str,
            "raw_trace_excerpt": str,
        }
        """
        steps = self._extract_steps(trace)
        if not steps:
            return self._build_result("other", [], "No steps found in trace.", "")

        # Run each detector in priority order and return the first hit
        for detector in [
            self._detect_repeated_action,
            self._detect_circular_loop,
            self._detect_context_truncation,
            self._detect_tool_misuse,
            self._detect_incorrect_reasoning,
        ]:
            result = detector(steps)
            if result is not None:
                result["raw_trace_excerpt"] = self._excerpt(steps, result["failed_steps"])
                return result

        # Default: unknown failure
        return self._build_result(
            "other",
            list(range(1, len(steps) + 1)),
            "Task ran to max steps without completing.",
            self._excerpt(steps, []),
        )

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------

    def _detect_repeated_action(self, steps: List[Dict]) -> Optional[Dict]:
        action_counts: Counter = Counter()
        action_positions: Dict[str, List[int]] = {}

        for s in steps:
            action = self._normalise_action(s["action"])
            action_counts[action] += 1
            action_positions.setdefault(action, []).append(s["step"])

        for action, count in action_counts.items():
            if count >= 3:
                failed_steps = action_positions[action]
                return self._build_result(
                    "repeated_action",
                    failed_steps,
                    f"The agent repeated the action '{action[:80]}' {count} times without progress.",
                )
        return None

    def _detect_circular_loop(self, steps: List[Dict]) -> Optional[Dict]:
        actions = [self._normalise_action(s["action"]) for s in steps]
        n = len(actions)

        # Check for repeated sub-sequences of length 2..5
        for seq_len in range(2, 6):
            if n < seq_len * 2:
                continue
            for start in range(n - seq_len * 2 + 1):
                seq = actions[start : start + seq_len]
                rest = actions[start + seq_len :]
                # Look for a repeat of seq in rest
                for i in range(len(rest) - seq_len + 1):
                    if rest[i : i + seq_len] == seq:
                        failed_start = steps[start]["step"]
                        failed_end = steps[start + seq_len * 2 - 1]["step"]
                        return self._build_result(
                            "circular_loop",
                            list(range(failed_start, failed_end + 1)),
                            f"The agent repeated a sequence of {seq_len} actions in a loop.",
                        )
        return None

    def _detect_context_truncation(self, steps: List[Dict]) -> Optional[Dict]:
        failed_steps = []
        for s in steps:
            obs = s.get("observation", "")
            if "[TRUNCATED" in obs or len(obs) / CHARS_PER_TOKEN > MAX_OBS_TOKENS:
                failed_steps.append(s["step"])

        if failed_steps:
            return self._build_result(
                "context_truncation",
                failed_steps,
                f"Observations in steps {failed_steps} exceeded the context window limit, causing loss of information.",
            )
        return None

    def _detect_tool_misuse(self, steps: List[Dict]) -> Optional[Dict]:
        failed_steps = []
        for s in steps:
            obs = s.get("observation", "").lower()
            # Convert action to string if it's a dict
            action = s.get("action", "")
            action_str = str(action).lower() if isinstance(action, dict) else action.lower()
            if obs.startswith("error: could not parse") or obs.startswith("error: tool") or \
               ("error" in obs and ("argument" in obs or "syntax" in obs or "format" in obs)):
                failed_steps.append(s["step"])
            elif "error" in obs and not s.get("success", True):
                # Check if the action itself looks malformed
                if not re.search(r"\w+\(", action_str):
                    failed_steps.append(s["step"])

        if len(failed_steps) >= 2:
            return self._build_result(
                "tool_misuse",
                failed_steps,
                f"The agent produced malformed tool calls or received unexpected errors at steps {failed_steps}.",
            )
        return None

    def _detect_incorrect_reasoning(self, steps: List[Dict]) -> Optional[Dict]:
        """
        Heuristic: a step's thought contradicts a prior successful observation
        by repeating an action whose previous observation indicated success.
        """
        seen_successful_outcomes: Dict[str, str] = {}
        failed_steps = []

        for s in steps:
            action = self._normalise_action(s["action"])
            obs = s.get("observation", "")
            thought = s.get("thought", "")

            if action in seen_successful_outcomes:
                prior_obs = seen_successful_outcomes[action]
                # If the prior observation confirmed success but agent re-tries the same action
                if any(kw in prior_obs.lower() for kw in ["success", "done", "written", "created", "ok"]):
                    failed_steps.append(s["step"])

            if s.get("success", False) or (obs and not obs.lower().startswith("error")):
                seen_successful_outcomes[action] = obs

        if failed_steps:
            return self._build_result(
                "incorrect_reasoning",
                failed_steps,
                f"The agent repeated actions at steps {failed_steps} that had already succeeded, "
                "suggesting incorrect reasoning about task state.",
            )
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_steps(trace: Any) -> List[Dict]:
        if hasattr(trace, "steps"):
            return [
                {
                    "step": s.step,
                    # Support both base_agent.TraceStep (thought) and trace_logger.TraceStep (reasoning)
                    "thought": getattr(s, "thought", getattr(s, "reasoning", "")),
                    "action": s.action,
                    "observation": s.observation,
                    "success": getattr(s, "success", True),
                }
                for s in trace.steps
            ]
        if isinstance(trace, dict) and "steps" in trace:
            return trace["steps"]
        if isinstance(trace, list):
            return trace
        return []

    @staticmethod
    def _normalise_action(action: Any) -> str:
        # Handle both dict and string action formats
        if isinstance(action, dict):
            # Convert dict to string representation
            action = str(action)
        action_str = str(action).strip().lower()
        return action_str

    @staticmethod
    def _build_result(
        failure_type: str,
        failed_steps: List[int],
        pattern_summary: str,
        raw_trace_excerpt: str = "",
    ) -> Dict[str, Any]:
        return {
            "failure_type": failure_type,
            "failed_steps": failed_steps,
            "pattern_summary": pattern_summary,
            "raw_trace_excerpt": raw_trace_excerpt,
        }

    @staticmethod
    def _excerpt(steps: List[Dict], highlight_steps: List[int]) -> str:
        lines = []
        for s in steps[-10:]:  # last 10 steps
            marker = " <<" if s["step"] in highlight_steps else ""
            obs = s.get("observation", "")
            obs_short = (obs[:150] + "...") if len(obs) > 150 else obs
            # Handle both string and dict action formats
            action = s.get("action", "")
            action_str = str(action) if isinstance(action, dict) else action
            lines.append(
                f"Step {s['step']}{marker}\n"
                f"  Thought: {s.get('thought', '')[:100]}\n"
                f"  Action: {action_str[:100]}\n"
                f"  Observation: {obs_short}"
            )
        return "\n".join(lines)
