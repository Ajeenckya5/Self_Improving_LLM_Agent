"""Experiment runner: baseline vs strategy-enhanced agent."""

import random
import shutil
from pathlib import Path
from typing import Any

from tasks import get_all_tasks
from tasks.base import Task
from environment.controlled import ControlledEnvironment
from tracing.logger import TraceLogger, ExecutionTrace
from agent.baseline import BaselineAgent
from agent.strategy_enhanced import StrategyEnhancedAgent
from failure_analysis.analyzer import FailureAnalyzer
from strategy_memory.store import StrategyMemory
from .metrics import ExperimentMetrics

try:
    from config import USE_STUDENT_ANALYZER, STUDENT_ADAPTER_PATH
except ImportError:
    USE_STUDENT_ANALYZER = False
    STUDENT_ADAPTER_PATH = None


class ExperimentRunner:
    """Runs experiments comparing baseline and strategy-enhanced agents."""

    def __init__(
        self,
        env_root: Path | str,
        traces_dir: Path | str,
        strategy_memory: StrategyMemory | None = None,
        max_steps: int = 15,
        num_attempts: int = 1,  # Attempts per task per agent
        seed: int = 42,
    ):
        self.env_root = Path(env_root)
        self.traces_dir = Path(traces_dir)
        self.strategy_memory = strategy_memory or StrategyMemory()
        self.max_steps = max_steps
        self.num_attempts = num_attempts
        self.seed = seed
        self.metrics = ExperimentMetrics()

    def _fresh_env(self, task: Task) -> tuple[Task, ControlledEnvironment]:
        """Create fresh env for task (re-setup)."""
        task_root = self.env_root / task.task_id
        shutil.rmtree(task_root, ignore_errors=True)
        task_root.mkdir(parents=True, exist_ok=True)
        task.env_root = str(task_root)
        task.setup()
        env = ControlledEnvironment(str(task_root), task)
        return task, env

    def run_single(
        self,
        task: Task,
        agent_type: str,
        attempt: int,
        strategies: list[str] | None = None,
    ) -> tuple[bool, ExecutionTrace | None]:
        """Run one task with one agent, return (success, trace)."""
        task, env = self._fresh_env(task)
        trace_logger = TraceLogger(self.traces_dir)

        if agent_type == "baseline":
            agent = BaselineAgent(max_steps=self.max_steps)
        else:
            agent = StrategyEnhancedAgent(max_steps=self.max_steps)

        result = agent.run(
            task=task,
            env=env,
            trace_logger=trace_logger,
            attempt=attempt,
            strategies=strategies,
        )

        # Run verifier
        state = env.get_state()
        verify_result = task.verify(state)
        success = verify_result.success

        if result.trace:
            result.trace.success = success
            result.trace.final_message = verify_result.message
            trace_logger.save_trace(result.trace)

        # On failure: analyze and store strategy
        # Use QLoRA student model if available, else fall back to gpt-4o-mini
        if not success and result.trace:
            if USE_STUDENT_ANALYZER and STUDENT_ADAPTER_PATH and STUDENT_ADAPTER_PATH.exists():
                from distillation.student_analyzer import StudentFailureAnalyzer
                analyzer = StudentFailureAnalyzer(adapter_path=STUDENT_ADAPTER_PATH)
            else:
                analyzer = FailureAnalyzer()
            analysis = analyzer.analyze(result.trace, task.description)
            self.strategy_memory.add(
                task_id=task.task_id,
                task_description=task.description,
                failure_category=analysis.failure_category or "unknown",
                corrective_strategy=analysis.corrective_strategy,
            )

        return success, result.trace

    def run_experiment(self) -> dict[str, Any]:
        """Run full experiment: baseline first, then strategy-enhanced."""
        random.seed(self.seed)
        tasks = get_all_tasks(str(self.env_root))

        results_baseline = []
        results_strategy = []

        for task in tasks:
            for attempt in range(self.num_attempts):
                # Baseline
                succ, trace = self.run_single(task, "baseline", attempt)
                results_baseline.append({
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "success": succ,
                    "trace": trace,
                })

        # Strategy-enhanced: retrieve strategies before each task
        for task in tasks:
            strategies = self.strategy_memory.retrieve(task.description, top_k=3)
            for attempt in range(self.num_attempts):
                succ, trace = self.run_single(
                    task, "strategy_enhanced", attempt, strategies=strategies
                )
                results_strategy.append({
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "success": succ,
                    "trace": trace,
                })

        self.metrics.record(results_baseline, results_strategy)
        return self.metrics.to_dict()
