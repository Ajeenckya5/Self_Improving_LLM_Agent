"""Dataset collector for organizing labeled execution traces."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..analysis.failure_analyzer import FailureAnalyzer
from ..utils.logger import get_logger
from ..utils.trace_logger import ExecutionTrace, TraceLogger

logger = get_logger(__name__)


@dataclass
class LabeledTrace:
    """Execution trace with failure analysis label."""
    task_id: str
    agent_type: str
    attempt: int
    horizon: int
    success: bool
    failure_type: Optional[str] = None
    failed_steps: Optional[List[int]] = None
    pattern_summary: Optional[str] = None
    trace_file: Optional[str] = None
    num_steps: int = 0
    elapsed_s: float = 0.0
    strategies_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetCollector:
    """
    Collects execution traces, analyzes failures, and creates a labeled dataset.
    Supports CSV export and trace file organization.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        traces_dir: str | Path | None = None,
        analyzer: Optional[FailureAnalyzer] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.traces_dir = Path(traces_dir) if traces_dir else self.dataset_dir / "traces"
        self.analyzer = analyzer or FailureAnalyzer()

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)

        self.labeled_traces: List[LabeledTrace] = []

    def add_trace(
        self,
        trace: ExecutionTrace,
        horizon: int,
        elapsed_s: float = 0.0,
        strategies_used: int = 0,
        save_trace: bool = True,
    ) -> LabeledTrace:
        """
        Process a trace: analyze failures, save to disk, and add to dataset.

        Returns
        -------
        LabeledTrace with failure type and related metadata.
        """
        # Analyze failure if task failed
        failure_type: Optional[str] = None
        failed_steps: Optional[List[int]] = None
        pattern_summary: Optional[str] = None

        if not trace.success:
            try:
                analysis = self.analyzer.analyze(trace)
                failure_type = analysis.get("failure_type", "other")
                failed_steps = analysis.get("failed_steps")
                pattern_summary = analysis.get("pattern_summary")
            except Exception as exc:
                logger.warning("Failure analysis failed for trace %s: %s", trace.task_id, exc)
                failure_type = "analysis_error"

        # Save trace to disk if requested
        trace_file: Optional[str] = None
        if save_trace:
            try:
                trace_path = self.traces_dir / f"{trace.task_id}_{trace.agent_type}_{trace.attempt}.json"
                trace_path.write_text(json.dumps(trace.to_dict(), indent=2))
                trace_file = str(trace_path.relative_to(self.dataset_dir))
            except Exception as exc:
                logger.warning("Failed to save trace: %s", exc)

        # Create labeled trace
        labeled = LabeledTrace(
            task_id=trace.task_id,
            agent_type=trace.agent_type,
            attempt=trace.attempt,
            horizon=horizon,
            success=trace.success,
            failure_type=failure_type,
            failed_steps=failed_steps,
            pattern_summary=pattern_summary,
            trace_file=trace_file,
            num_steps=len(trace.steps),
            elapsed_s=elapsed_s,
            strategies_used=strategies_used,
        )

        self.labeled_traces.append(labeled)
        return labeled

    def save_dataset_csv(self, filename: str = "labeled_dataset.csv") -> Path:
        """
        Export all labeled traces to a CSV file.

        Returns
        -------
        Path to the saved CSV file.
        """
        if not self.labeled_traces:
            logger.warning("No labeled traces to save")
            return self.dataset_dir / filename

        csv_path = self.dataset_dir / filename

        # Get all keys from the first trace
        fieldnames = list(asdict(self.labeled_traces[0]).keys())

        # Handle list fields (failed_steps)
        def serialize_value(v):
            if isinstance(v, list):
                return json.dumps(v)
            return v

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trace in self.labeled_traces:
                row = {k: serialize_value(v) for k, v in asdict(trace).items()}
                writer.writerow(row)

        logger.info("Saved %d labeled traces to %s", len(self.labeled_traces), csv_path)
        return csv_path

    def save_dataset_jsonl(self, filename: str = "labeled_dataset.jsonl") -> Path:
        """
        Export all labeled traces to a JSONL file (one trace per line).

        Returns
        -------
        Path to the saved JSONL file.
        """
        if not self.labeled_traces:
            logger.warning("No labeled traces to save")
            return self.dataset_dir / filename

        jsonl_path = self.dataset_dir / filename

        with open(jsonl_path, "w") as f:
            for trace in self.labeled_traces:
                f.write(json.dumps(trace.to_dict()) + "\n")

        logger.info("Saved %d labeled traces to %s", len(self.labeled_traces), jsonl_path)
        return jsonl_path

    def get_failure_statistics(self) -> Dict[str, int]:
        """
        Get counts of each failure type.

        Returns
        -------
        Dict mapping failure type to count.
        """
        stats: Dict[str, int] = {}
        for trace in self.labeled_traces:
            if trace.failure_type:
                stats[trace.failure_type] = stats.get(trace.failure_type, 0) + 1
        return stats

    def filter_by_agent_type(self, agent_type: str) -> List[LabeledTrace]:
        """Get all labeled traces for a specific agent type."""
        return [t for t in self.labeled_traces if t.agent_type == agent_type]

    def filter_by_failure_type(self, failure_type: str) -> List[LabeledTrace]:
        """Get all labeled traces with a specific failure type."""
        return [t for t in self.labeled_traces if t.failure_type == failure_type]

    def filter_by_horizon(self, horizon: int) -> List[LabeledTrace]:
        """Get all labeled traces for a specific horizon."""
        return [t for t in self.labeled_traces if t.horizon == horizon]

    def get_success_rate(self) -> float:
        """Get overall success rate across all traces."""
        if not self.labeled_traces:
            return 0.0
        successes = sum(1 for t in self.labeled_traces if t.success)
        return successes / len(self.labeled_traces)

    def get_success_rate_by_agent(self) -> Dict[str, float]:
        """Get success rate per agent type."""
        agents = {}
        for trace in self.labeled_traces:
            if trace.agent_type not in agents:
                agents[trace.agent_type] = {"success": 0, "total": 0}
            agents[trace.agent_type]["total"] += 1
            if trace.success:
                agents[trace.agent_type]["success"] += 1

        return {
            agent: stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
            for agent, stats in agents.items()
        }

    def get_success_rate_by_horizon(self) -> Dict[int, float]:
        """Get success rate per horizon."""
        horizons = {}
        for trace in self.labeled_traces:
            if trace.horizon not in horizons:
                horizons[trace.horizon] = {"success": 0, "total": 0}
            horizons[trace.horizon]["total"] += 1
            if trace.success:
                horizons[trace.horizon]["success"] += 1

        return {
            h: stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
            for h, stats in horizons.items()
        }
