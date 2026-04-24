"""Prompt and JSON helpers shared by Member 2 analysis components."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from self_improving_agent/prompts."""
    path = PROMPTS_DIR / prompt_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, values: Dict[str, Any]) -> str:
    """
    Render simple {placeholder} templates without interpreting literal JSON
    braces in prompt examples.
    """
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse the first JSON object from an LLM response."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response")

    data = json.loads(match.group())
    if not isinstance(data, dict):
        raise ValueError("JSON response is not an object")
    return data


def coerce_int_list(value: Any) -> List[int]:
    """Return a stable list of integer step ids from messy model output."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(x) for x in re.findall(r"\d+", value)]
    if isinstance(value, Iterable):
        out: List[int] = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    return []


def trace_to_text(trace: Any, max_steps: int = 30, obs_chars: int = 600) -> str:
    """Convert an AgentTrace, dict trace, or step list into compact text."""
    if hasattr(trace, "to_text"):
        return trace.to_text(max_steps=max_steps)

    if hasattr(trace, "steps"):
        steps = trace.steps
    elif isinstance(trace, dict):
        steps = trace.get("steps", [])
    elif isinstance(trace, list):
        steps = trace
    else:
        steps = []

    lines: List[str] = []
    for raw_step in list(steps)[:max_steps]:
        if isinstance(raw_step, dict):
            step = raw_step.get("step", "?")
            thought = raw_step.get("thought", "")
            action = raw_step.get("action", "")
            observation = raw_step.get("observation", "")
            success = raw_step.get("success", False)
        else:
            step = getattr(raw_step, "step", "?")
            thought = getattr(raw_step, "thought", "")
            action = getattr(raw_step, "action", "")
            observation = getattr(raw_step, "observation", "")
            success = getattr(raw_step, "success", False)

        obs = str(observation)
        if len(obs) > obs_chars:
            obs = obs[:obs_chars] + "... [trimmed]"
        lines.append(
            f"Step {step} | success={success}\n"
            f"Thought: {thought}\n"
            f"Action: {action}\n"
            f"Observation: {obs}"
        )
    return "\n\n".join(lines)
