"""
End-to-end evaluation loop.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from tqdm import tqdm

from ..agent.base_agent import BaseAgent, AgentTrace
from ..analysis.failure_analyzer import FailureAnalyzer
from ..analysis.strategy_generator import StrategyGenerator
from ..memory.strategy_memory import StrategyMemory
from ..memory.retriever import Retriever
from ..utils.logger import get_logger

logger = get_logger(__name__)


def run_experiment(
    agent_class: Type[BaseAgent],
    tasks: List[Dict[str, Any]],
    config: Dict[str, Any],
    use_memory: bool = True,
    memory: Optional[StrategyMemory] = None,
    retriever: Optional[Retriever] = None,
    llm_client=None,
    label: str = "experiment",
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Run an end-to-end evaluation for one agent class on a list of tasks.

    Parameters
    ----------
    agent_class  : Agent class (BaseAgent, PlanActAgent, StrategyAgent)
    tasks        : List of task dicts (each must have 'id', 'description', 'horizon')
    config       : Loaded YAML config dict
    use_memory   : Whether to retrieve/store strategies
    memory       : Optional pre-built StrategyMemory instance
    retriever    : Optional pre-built Retriever instance
    llm_client   : Optional shared LLMClient
    label        : Short name for this experiment (used in results)
    dry_run      : If True, only run the first 5 tasks

    Returns
    -------
    DataFrame with one row per task.
    """
    if dry_run:
        tasks = tasks[:5]
        logger.info("[DRY RUN] Running only first 5 tasks.")

    # Build shared components
    if use_memory and memory is None:
        db_path = config.get("memory", {}).get("db_path", "results/strategy_memory.db")
        memory = StrategyMemory(db_path=db_path)
    if use_memory and retriever is None and memory is not None:
        retriever = Retriever(config=config, memory=memory)

    analyzer = FailureAnalyzer()
    generator = StrategyGenerator(config=config, llm_client=llm_client)

    top_k = config.get("memory", {}).get("top_k", 3)
    threshold = config.get("memory", {}).get("similarity_threshold", 0.65)

    results = []

    for task in tqdm(tasks, desc=label, unit="task"):
        task_id = task.get("id", "unknown")
        task_description = task.get("description", "")
        horizon = task.get("horizon", 0)

        # Retrieve strategies
        strategies = []
        used_strategy_ids = []

        if use_memory and retriever is not None:
            try:
                query_emb = retriever.embed(task_description)
                strategies = retriever.retrieve(query_emb, top_k=top_k, threshold=threshold)
                used_strategy_ids = [s["id"] for s in strategies]
            except Exception as exc:
                logger.warning("Retrieval failed for task %s: %s", task_id, exc)

        # Build and run agent
        agent_kwargs: Dict[str, Any] = {"config": config}
        if llm_client is not None:
            agent_kwargs["llm_client"] = llm_client
        if hasattr(agent_class, "__init__") and "retriever" in agent_class.__init__.__code__.co_varnames:
            agent_kwargs["retriever"] = retriever

        agent = agent_class(**agent_kwargs)

        t0 = time.time()
        try:
            success, trace = agent.run(task, strategies=strategies)
        except Exception as exc:
            logger.error("Agent crashed on task %s: %s", task_id, exc)
            success = False
            trace = AgentTrace(task_id=task_id, task_description=task_description)

        elapsed = time.time() - t0

        # Analyse failure and update memory
        failure_type = None
        if not success and use_memory and memory is not None and retriever is not None:
            try:
                failure = analyzer.analyze(trace)
                failure_type = failure.get("failure_type", "other")
                strategy_result = generator.generate(task, failure)
                query_emb = retriever.embed(task_description)
                # Merge tags from failure + strategy result
                combined_failure = {**failure, "tags": strategy_result.get("tags", [])}
                memory.store(
                    task_description=task_description,
                    failure_analysis=combined_failure,
                    strategy_text=strategy_result.get("strategy_text", ""),
                    embedding=query_emb,
                )
            except Exception as exc:
                logger.warning("Memory update failed for task %s: %s", task_id, exc)
        elif not success:
            try:
                failure = analyzer.analyze(trace)
                failure_type = failure.get("failure_type", "other")
            except Exception:
                failure_type = "other"

        # Update strategy outcomes
        if success and used_strategy_ids and memory is not None:
            for sid in used_strategy_ids:
                try:
                    memory.update_outcome(sid, success=True)
                except Exception:
                    pass

        results.append({
            "task_id": task_id,
            "horizon": horizon,
            "agent_type": agent_class.AGENT_TYPE,
            "success": success,
            "steps_taken": trace.total_steps,
            "failure_type": failure_type if not success else None,
            "strategies_used": len(used_strategy_ids),
            "elapsed_s": round(elapsed, 2),
            "label": label,
        })

        logger.info(
            "[%s] task=%s horizon=%d success=%s steps=%d",
            label, task_id, horizon, success, trace.total_steps,
        )

        # Cleanup environment
        env = task.get("env")
        if env is not None and hasattr(env, "cleanup"):
            try:
                env.cleanup()
            except Exception:
                pass

    return pd.DataFrame(results)
