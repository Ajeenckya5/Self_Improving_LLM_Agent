"""
Experiment runner for controlled filesystem/database tasks.

Bridges the structured Task suite (tasks/) with my agent implementations,
using ControlledEnvironment as the sandbox and my StrategyMemory for improvement.

Usage:
    from self_improving_agent.experiments.controlled_runner import run_controlled_experiment
    results = run_controlled_experiment(config, env_root="sandbox/")
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..agent.base_agent import BaseAgent
from ..agent.plan_act_agent import PlanActAgent
from ..agent.strategy_agent import StrategyAgent
from ..analysis.failure_analyzer import FailureAnalyzer
from ..analysis.strategy_generator import StrategyGenerator
from ..environments.controlled_env import ControlledEnvironment
from ..memory.retriever import Retriever
from ..memory.strategy_memory import StrategyMemory
from ..tasks.base import Task
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tool description injected into agent prompts for controlled environments
# ---------------------------------------------------------------------------

_CONTROLLED_TOOLS_DESC = """Available tools (controlled environment):
  list_dir(path)           - list files/directories at path
  read_file(path)          - read a file's contents
  write_file(path, content)- write content to a file (creates or overwrites)
  create_dir(path)         - create a directory (with parents)
  delete_file(path)        - delete a file
  move_file(src, dst)      - move/rename a file
  execute_sql(sql)         - execute a SQL statement (CREATE, INSERT, UPDATE, etc.)
  query_sql(sql)           - execute a SELECT and return results
  list_tables()            - list all tables in the database
  get_schema(table)        - show schema of a table
  finish(result)           - declare the task complete"""


# ---------------------------------------------------------------------------
# Agent subclasses with controlled-env prompt overrides
# ---------------------------------------------------------------------------

class _ControlledPromptMixin:
    """Mixin that replaces the system/user prompts for controlled-env tasks."""

    def _build_system_prompt(self, strategies):
        return (
            "You are a capable AI agent solving tasks step by step.\n\n"
            "Format EVERY response as:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool_name>(<arguments>)\n\n"
            f"{_CONTROLLED_TOOLS_DESC}\n\n"
            "Rules:\n"
            "- Always explore the environment first (list_dir or list_tables).\n"
            "- Think carefully before each action.\n"
            "- Do not repeat a failed action without changing it.\n"
            "- Call finish() when the task is complete."
        )

    def _build_user_prompt(self, task):
        return (
            f"Task: {task.get('description', '')}\n\n"
            "Start by exploring the environment to understand its current state, "
            "then execute the task step by step."
        )


class ControlledBaseAgent(_ControlledPromptMixin, BaseAgent):
    AGENT_TYPE = "react"


class ControlledPlanActAgent(_ControlledPromptMixin, PlanActAgent):
    AGENT_TYPE = "plan_act"


class ControlledStrategyAgent(_ControlledPromptMixin, StrategyAgent):
    AGENT_TYPE = "strategy"

    def _build_system_prompt(self, strategies):
        base = super()._build_system_prompt(strategies)
        if strategies:
            strats_text = "\n".join(
                f"{i+1}. {s.get('strategy_text', '').strip()}"
                for i, s in enumerate(strategies)
            )
            base += (
                f"\n\nBased on past experience with similar tasks, apply these strategies:\n"
                f"{strats_text}\n\nUse these proactively to avoid repeating past mistakes."
            )
        return base


# ---------------------------------------------------------------------------
# Environment adapter
# ---------------------------------------------------------------------------

# Maps tool names to their positional argument names (in order)
_TOOL_ARG_NAMES: dict[str, list[str]] = {
    "list_dir": ["path"],
    "read_file": ["path"],
    "write_file": ["path", "content"],
    "create_dir": ["path"],
    "delete_file": ["path"],
    "move_file": ["src", "dst"],
    "execute_sql": ["sql"],
    "query_sql": ["sql"],
    "list_tables": [],
    "get_schema": ["table"],
}


class ControlledEnvAdapter:
    """
    Wraps ControlledEnvironment to implement the env.step() / env.is_success()
    interface expected by BaseAgent._execute_action().
    """

    def __init__(self, task: Task, env: ControlledEnvironment):
        self._task = task
        self._env = env

    def step(self, action_str: str) -> str:
        """Parse an agent action string and dispatch to ControlledEnvironment."""
        action_str = action_str.strip()
        # Strip leading "Action:" prefix
        if action_str.lower().startswith("action:"):
            action_str = action_str[7:].strip()

        if action_str.lower().startswith("finish"):
            return "Task declared complete."

        # Match tool_name(args)
        m = re.match(r"(\w+)\((.*)\)$", action_str, re.DOTALL)
        if not m:
            return (
                "Error: Could not parse action. "
                "Expected format: tool_name(arguments) — e.g. read_file(path/to/file)"
            )

        tool_name = m.group(1).strip()
        args_str = m.group(2).strip()

        kwargs = self._parse_args(tool_name, args_str)
        return self._env.execute(tool_name, **kwargs)

    def is_success(self) -> bool:
        state = self._env.get_state()
        result = self._task.verify(state)
        return result.success

    def cleanup(self) -> None:
        pass  # Env lifetime managed by the runner

    # ------------------------------------------------------------------

    def _parse_args(self, tool_name: str, args_str: str) -> dict[str, str]:
        arg_names = _TOOL_ARG_NAMES.get(tool_name, [])
        if not arg_names or not args_str:
            return {}

        # Try keyword-arg style first: key=value, key=value
        if re.search(r"\w+=", args_str):
            kwargs: dict[str, str] = {}
            # Split carefully: keyword args for multi-arg tools
            # For single-arg tools, skip this branch to avoid false positives in SQL
            if len(arg_names) > 1:
                try:
                    for pair in re.split(r",\s*(?=\w+=)", args_str):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            kwargs[k.strip()] = v.strip().strip('"').strip("'")
                    if kwargs and all(k in arg_names for k in kwargs):
                        return kwargs
                except Exception:
                    pass

        # Positional: split by first N-1 commas (preserves commas in last arg, e.g. SQL)
        n_splits = len(arg_names) - 1
        if n_splits == 0:
            parts = [args_str]
        else:
            parts = args_str.split(",", n_splits)

        result = {}
        for i, name in enumerate(arg_names):
            if i < len(parts):
                result[name] = parts[i].strip().strip('"').strip("'")
        return result


# ---------------------------------------------------------------------------
# Task setup helpers
# ---------------------------------------------------------------------------

def _make_task_dict(
    task: Task,
    env_root: Path,
    horizon: int = 10,
) -> Tuple[Dict[str, Any], ControlledEnvAdapter]:
    """
    Setup a fresh environment for a task and return (task_dict, adapter).
    task_dict is the format expected by BaseAgent.run().
    """
    task_root = env_root / task.task_id
    shutil.rmtree(task_root, ignore_errors=True)
    task_root.mkdir(parents=True, exist_ok=True)
    task.env_root = str(task_root)
    task.setup()

    env = ControlledEnvironment(str(task_root), task)
    adapter = ControlledEnvAdapter(task, env)

    task_dict = {
        "id": task.task_id,
        "description": task.description,
        "horizon": horizon,
        "env": adapter,
    }
    return task_dict, adapter


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_controlled_experiment(
    config: Dict[str, Any],
    env_root: str = "sandbox",
    results_dir: str = "results",
    num_attempts: int = 3,
    horizons: Optional[List[int]] = None,
    llm_client: Optional[LLMClient] = None,
    dry_run: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Run the three-condition experiment on the controlled task suite:
      1. ReAct baseline
      2. Plan-and-Act baseline
      3. Self-Improving (strategy-guided, memory accumulates)

    horizons: step limits to sweep over; defaults to config evaluation.horizons.
    Returns a dict of {condition_label: DataFrame}.
    """
    env_root_path = Path(env_root)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    active_horizons: List[int] = (
        horizons if horizons is not None
        else list(config.get("evaluation", {}).get("horizons", [5, 10, 15, 20]))
    )

    if llm_client is None:
        llm_client = LLMClient(config)

    analyzer = FailureAnalyzer()
    generator = StrategyGenerator(config=config, llm_client=llm_client)

    top_k = config.get("memory", {}).get("top_k", 3)
    threshold = config.get("memory", {}).get("similarity_threshold", 0.65)

    # Strategy memory for the self-improving condition
    db_path = str(results_path / "strategy_memory_controlled.db")
    memory = StrategyMemory(db_path=db_path)
    retriever = Retriever(config=config, memory=memory)

    checkpoint_path = results_path / "controlled_results_checkpoint.csv"
    checkpoint_written = False

    def _get_tasks():
        from ..tasks import get_all_tasks as _gat
        return _gat(str(env_root_path))

    conditions = [
        ("ReAct", ControlledBaseAgent, False),
        ("Plan-and-Act", ControlledPlanActAgent, False),
        ("Self-Improving (ours)", ControlledStrategyAgent, True),
    ]

    all_rows: List[Dict[str, Any]] = []
    all_results: Dict[str, pd.DataFrame] = {}

    for label, agent_cls, use_memory in conditions:
        logger.info("Running condition: %s", label)
        tasks = _get_tasks()
        if dry_run:
            tasks = tasks[:3]
            horizons_run = active_horizons[:2]
            logger.info("[DRY RUN] 3 tasks, 2 horizons.")
        else:
            horizons_run = active_horizons

        rows = []
        for horizon in horizons_run:
            logger.info("  horizon=%d", horizon)
            for attempt_idx in range(num_attempts):
                for task in tasks:
                    task_dict, adapter = _make_task_dict(task, env_root_path, horizon=horizon)

                    strategies: List[Dict[str, Any]] = []
                    used_ids: List[int] = []
                    if use_memory:
                        try:
                            q_emb = retriever.embed(task.description)
                            strategies = retriever.retrieve(q_emb, top_k=top_k, threshold=threshold)
                            used_ids = [s["id"] for s in strategies]
                        except Exception as exc:
                            logger.warning("Strategy retrieval failed: %s", exc)

                    agent_kwargs: Dict[str, Any] = {"config": config, "llm_client": llm_client}
                    if use_memory:
                        agent_kwargs["retriever"] = retriever
                    agent = agent_cls(**agent_kwargs)
                    # Enforce horizon as actual step budget
                    agent.max_steps = horizon

                    t0 = time.time()
                    try:
                        success, trace = agent.run(task_dict, strategies=strategies)
                    except Exception as exc:
                        logger.error("Agent crashed on task %s: %s", task.task_id, exc)
                        success = False
                        trace = None

                    try:
                        success = adapter.is_success()
                    except Exception:
                        pass

                    elapsed = time.time() - t0
                    steps_taken = trace.total_steps if trace else 0

                    failure_type: Optional[str] = None
                    if not success:
                        try:
                            fa = analyzer.analyze(trace)
                            failure_type = fa.get("failure_type", "other")
                            if use_memory and trace is not None:
                                sg = generator.generate(
                                    {"description": task.description, "id": task.task_id},
                                    fa,
                                )
                                q_emb = retriever.embed(task.description)
                                memory.store(
                                    task_description=task.description,
                                    failure_analysis={**fa, "tags": sg.get("tags", [])},
                                    strategy_text=sg.get("strategy_text", ""),
                                    embedding=q_emb,
                                )
                        except Exception as exc:
                            logger.warning("Post-failure analysis failed: %s", exc)

                    if success and used_ids:
                        for sid in used_ids:
                            try:
                                memory.update_outcome(sid, success=True)
                            except Exception:
                                pass

                    row = {
                        "task_id": task.task_id,
                        "horizon": horizon,
                        "agent_type": agent_cls.AGENT_TYPE,
                        "attempt": attempt_idx,
                        "success": success,
                        "steps_taken": steps_taken,
                        "failure_type": failure_type if not success else None,
                        "strategies_used": len(used_ids),
                        "elapsed_s": round(elapsed, 2),
                        "label": label,
                    }
                    rows.append(row)
                    all_rows.append(row)

                    # Checkpoint after every task
                    pd.DataFrame(all_rows).to_csv(
                        checkpoint_path, index=False,
                        mode="w" if not checkpoint_written else "w",
                    )
                    checkpoint_written = True

                    logger.info(
                        "[%s] h=%d attempt=%d task=%s success=%s steps=%d",
                        label, horizon, attempt_idx, task.task_id, success, steps_taken,
                    )

        df = pd.DataFrame(rows)
        all_results[label] = df

    return all_results
