"""Failure analyzer: rule-based checks + LLM classification."""

import json
import os
from dataclasses import dataclass
from typing import Any

from tracing.logger import ExecutionTrace
from .categories import FAILURE_CATEGORIES, CATEGORY_DESCRIPTIONS


@dataclass
class FailureAnalysis:
    """Result of failure analysis."""
    rule_based_findings: list[str]
    failure_category: str | None
    corrective_strategy: str
    raw_llm_response: str = ""


class FailureAnalyzer:
    """Analyzes failures using rules first, then LLM when needed."""

    def __init__(self, model: str = "gpt-4o-mini", openai_api_key: str | None = None):
        self.model = model
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")

    def analyze(self, trace: ExecutionTrace, task_description: str) -> FailureAnalysis:
        """Analyze a failed execution trace and produce a corrective strategy."""
        rule_findings = self._rule_based_checks(trace)

        # If rules give a clear signal, we can skip LLM for some cases
        # But we still want LLM to generate the corrective strategy
        category, strategy, raw = self._llm_analyze(
            trace, task_description, rule_findings
        )
        return FailureAnalysis(
            rule_based_findings=rule_findings,
            failure_category=category,
            corrective_strategy=strategy,
            raw_llm_response=raw,
        )

    def _rule_based_checks(self, trace: ExecutionTrace) -> list[str]:
        findings = []

        # Repeated actions
        actions = [s.action.get("tool") or s.action.get("action") for s in trace.steps]
        from collections import Counter
        counts = Counter(a for a in actions if a)
        for action, count in counts.items():
            if count >= 3:
                findings.append(f"Repeated action '{action}' {count} times (possible loop)")

        # Environment errors in observations
        for s in trace.steps:
            obs = s.observation or ""
            if "Error:" in obs or "error:" in obs or "does not exist" in obs:
                findings.append(f"Environment error in step {s.step}: {obs[:80]}...")
                break

        # Missing constraints: check if agent explored before acting
        if trace.steps:
            first_actions = [s.action.get("tool") or s.action.get("action") for s in trace.steps[:3]]
            if "list_dir" not in first_actions and "list_tables" not in first_actions and "get_schema" not in first_actions:
                findings.append("Agent may not have verified environment state before acting")

        # Max steps reached without done
        if trace.steps and len(trace.steps) >= 10:
            findings.append("Reached many steps without completing (possible inefficiency)")

        return findings

    def _llm_analyze(
        self,
        trace: ExecutionTrace,
        task_description: str,
        rule_findings: list[str],
    ) -> tuple[str | None, str, str]:
        """Use LLM to classify failure and generate corrective strategy."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._api_key)
        except ImportError:
            return None, "Install openai package for LLM analysis", ""

        # Summarize trace for prompt (include reasoning when available)
        steps_text = []
        for s in trace.steps[:15]:  # Limit to avoid token overflow
            act = s.action.get("tool") or s.action.get("action") or str(s.action)[:80]
            obs = (s.observation or "")[:200]
            reason = getattr(s, "reasoning", "") or ""
            step_line = f"Step {s.step}: action={act}\n  observation: {obs}"
            if reason:
                step_line += f"\n  reasoning: {reason[:150]}"
            steps_text.append(step_line)

        prompt = f"""A task failed. Analyze the execution trace and produce a corrective strategy.

Task: {task_description}

Rule-based findings: {rule_findings or 'None'}

Execution trace (last steps):
{chr(10).join(steps_text)}

Failure categories: {', '.join(FAILURE_CATEGORIES)}

Respond with JSON only:
{{
  "category": "one of the categories above",
  "corrective_strategy": "One specific, actionable rule for future attempts. Example: 'Verify schema before altering a database table' or 'Recheck file ownership before changing permissions'. Be concrete."
}}"""

        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            content = resp.choices[0].message.content.strip()
        except Exception as e:
            return None, f"LLM error: {e}", ""

        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            data = json.loads(content)
            cat = data.get("category")
            if cat not in FAILURE_CATEGORIES:
                cat = FAILURE_CATEGORIES[0]
            strategy = data.get("corrective_strategy", "Retry with careful planning")
            return cat, strategy, content
        except json.JSONDecodeError:
            return None, "Retry with careful planning; verify environment state first", content
