"""Reproducibility tracking for experiment runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RunMetadata:
    """Metadata for a single experiment run."""
    run_id: str  # Unique identifier for this run
    timestamp: str  # ISO 8601 format
    config_hash: str  # Hash of the config file
    model_profile: str  # e.g., "xai", "haiku", "groq"
    agent_types: list[str] = field(default_factory=list)  # ["react", "plan_act", "strategy"]
    horizons: list[int] = field(default_factory=list)  # e.g., [5, 10, 15, 20]
    n_tasks: int = 0
    dry_run: bool = False
    num_conditions: int = 0  # e.g., 3 (ReAct, Plan-and-Act, Self-Improving)
    dataset_path: Optional[str] = None
    traces_path: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class RunTracker:
    """
    Tracks experiment metadata for reproducibility.
    Generates unique run IDs and maintains run logs.
    """

    def __init__(self, results_dir: str | Path = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.runs_log_path = self.results_dir / "runs_log.jsonl"
        self.current_run: Optional[RunMetadata] = None

    def generate_run_id(self) -> str:
        """Generate a unique run ID based on timestamp."""
        timestamp = datetime.now().isoformat(timespec="seconds")
        run_id = f"run_{timestamp.replace(':', '').replace('-', '')}"
        return run_id

    def compute_config_hash(self, config_dict: Dict[str, Any]) -> str:
        """Compute a hash of the configuration for reproducibility tracking."""
        config_json = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]

    def start_run(
        self,
        config: Dict[str, Any],
        horizons: list[int],
        n_tasks: int,
        dry_run: bool = False,
        notes: str = "",
    ) -> RunMetadata:
        """
        Start a new experiment run and return metadata.

        Parameters
        ----------
        config : The loaded config dict
        horizons : List of task horizons
        n_tasks : Number of tasks per horizon
        dry_run : Whether this is a dry run
        notes : Optional notes about this run

        Returns
        -------
        RunMetadata instance for this run.
        """
        run_id = self.generate_run_id()
        config_hash = self.compute_config_hash(config)
        timestamp = datetime.now().isoformat()

        model_profile = config.get("active_profile", "unknown")
        agent_types = ["react", "plan_act", "strategy"]

        self.current_run = RunMetadata(
            run_id=run_id,
            timestamp=timestamp,
            config_hash=config_hash,
            model_profile=model_profile,
            agent_types=agent_types,
            horizons=horizons,
            n_tasks=n_tasks,
            dry_run=dry_run,
            num_conditions=3,
            notes=notes,
        )

        logger.info("Started run %s with config hash %s", run_id, config_hash)
        return self.current_run

    def end_run(
        self,
        dataset_path: Optional[str] = None,
        traces_path: Optional[str] = None,
    ) -> RunMetadata:
        """
        End the current run and log metadata.

        Parameters
        ----------
        dataset_path : Path to the labeled dataset CSV/JSONL
        traces_path : Path to the traces directory

        Returns
        -------
        RunMetadata instance for the completed run.
        """
        if self.current_run is None:
            raise RuntimeError("No active run to end")

        self.current_run.dataset_path = dataset_path
        self.current_run.traces_path = traces_path

        # Append to runs log
        with open(self.runs_log_path, "a") as f:
            f.write(self.current_run.to_json() + "\n")

        logger.info("Ended run %s", self.current_run.run_id)
        return self.current_run

    def get_current_run(self) -> Optional[RunMetadata]:
        """Get the current active run metadata."""
        return self.current_run

    def get_all_runs(self) -> list[RunMetadata]:
        """Load all historical runs from the log."""
        runs = []
        if self.runs_log_path.exists():
            with open(self.runs_log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            run = RunMetadata(**data)
                            runs.append(run)
                        except Exception as exc:
                            logger.warning("Failed to parse run log entry: %s", exc)
        return runs

    def save_config_snapshot(self, config: Dict[str, Any], suffix: str = "") -> Path:
        """
        Save a snapshot of the config for this run.

        Parameters
        ----------
        config : The configuration dict
        suffix : Optional suffix for the filename

        Returns
        -------
        Path to the saved config file.
        """
        if self.current_run is None:
            raise RuntimeError("No active run")

        filename = f"config_{self.current_run.run_id}{suffix}.yaml"
        config_path = self.results_dir / filename

        import yaml

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        logger.info("Saved config snapshot to %s", config_path)
        return config_path


class ReproducibilityHelper:
    """
    Combines run tracking and dataset collection for full reproducibility.
    Use this in experiment runners.
    """

    def __init__(self, results_dir: str | Path = "results"):
        self.results_dir = Path(results_dir)
        self.tracker = RunTracker(results_dir=self.results_dir)
        self.dataset_collectors: Dict[str, Any] = {}  # {agent_type: DatasetCollector}

    def start_experiment(
        self,
        config: Dict[str, Any],
        horizons: list[int],
        n_tasks: int,
        dry_run: bool = False,
        notes: str = "",
    ) -> str:
        """
        Start an experiment and return the run ID.

        Returns
        -------
        run_id for later reference.
        """
        run_metadata = self.tracker.start_run(
            config=config,
            horizons=horizons,
            n_tasks=n_tasks,
            dry_run=dry_run,
            notes=notes,
        )
        self.tracker.save_config_snapshot(config)

        # Create a run-specific directory
        run_dir = self.results_dir / run_metadata.run_id
        run_dir.mkdir(exist_ok=True)

        logger.info("Experiment started: %s", run_metadata.run_id)
        return run_metadata.run_id

    def end_experiment(self) -> RunMetadata:
        """End the experiment and finalize logging."""
        run_metadata = self.tracker.get_current_run()
        if run_metadata is None:
            raise RuntimeError("No active experiment")

        dataset_path = None
        traces_path = None

        # Collect paths from dataset collectors
        for agent_type, collector in self.dataset_collectors.items():
            try:
                csv_path = collector.save_dataset_csv(f"labeled_dataset_{agent_type}.csv")
                dataset_path = str(csv_path)
                traces_path = str(collector.traces_dir)
            except Exception as exc:
                logger.warning("Failed to save dataset for %s: %s", agent_type, exc)

        return self.tracker.end_run(dataset_path=dataset_path, traces_path=traces_path)

    def register_dataset_collector(self, agent_type: str, collector: Any) -> None:
        """Register a dataset collector for an agent type."""
        self.dataset_collectors[agent_type] = collector
