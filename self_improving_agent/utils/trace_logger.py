"""Trace logger that records (state, action, observation) trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class TraceStep:
    """Single step: state snapshot, action taken, observation received."""
    step: int
    state: dict[str, Any]
    action: dict[str, Any]
    observation: str
    reasoning: str = ""


@dataclass
class ExecutionTrace:
    """Full trajectory for one task attempt."""
    task_id: str
    agent_type: str
    attempt: int
    steps: list[TraceStep] = field(default_factory=list)
    success: bool | None = None
    final_message: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "agent_type": self.agent_type,
            "attempt": self.attempt,
            "success": self.success,
            "final_message": self.final_message,
            "steps": [
                {
                    "step": s.step,
                    "state": s.state,
                    "action": s.action,
                    "observation": s.observation,
                    "reasoning": s.reasoning,
                }
                for s in self.steps
            ],
        }


class TraceLogger:
    """Records and persists execution traces to disk."""

    def __init__(self, traces_dir: Path | str | None = None):
        self.traces_dir = Path(traces_dir) if traces_dir else Path("traces")
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self._current: ExecutionTrace | None = None

    def start_trace(self, task_id: str, agent_type: str, attempt: int) -> None:
        self._current = ExecutionTrace(
            task_id=task_id,
            agent_type=agent_type,
            attempt=attempt,
        )

    def log_step(
        self,
        state: dict,
        action: dict,
        observation: str,
        reasoning: str = "",
    ) -> None:
        if not self._current:
            return
        self._current.steps.append(
            TraceStep(
                step=len(self._current.steps) + 1,
                state=state,
                action=action,
                observation=observation,
                reasoning=reasoning,
            )
        )

    def end_trace(self, success: bool, message: str = "") -> ExecutionTrace | None:
        if not self._current:
            return None
        self._current.success = success
        self._current.final_message = message
        trace = self._current
        self._current = None
        return trace

    def save_trace(self, trace: ExecutionTrace, run_id: str = "") -> Path:
        fname = f"{trace.task_id}_{trace.agent_type}_{trace.attempt}"
        if run_id:
            fname += f"_{run_id}"
        fpath = self.traces_dir / f"{fname}.json"
        fpath.write_text(json.dumps(trace.to_dict(), indent=2))
        return fpath

    def load_trace(self, path: Path) -> ExecutionTrace:
        data = json.loads(path.read_text())
        trace = ExecutionTrace(
            task_id=data["task_id"],
            agent_type=data["agent_type"],
            attempt=data["attempt"],
            success=data.get("success"),
            final_message=data.get("final_message", ""),
        )
        for s in data.get("steps", []):
            trace.steps.append(
                TraceStep(
                    step=s["step"],
                    state=s["state"],
                    action=s["action"],
                    observation=s["observation"],
                    reasoning=s.get("reasoning", ""),
                )
            )
        return trace
