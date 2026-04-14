"""
Plan-and-Act agent: first generates a high-level plan, then executes each step.
Extends BaseAgent with an upfront planning phase.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import AgentTrace, BaseAgent, TraceStep
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PlanActAgent(BaseAgent):
    """
    Plan-and-Act variant:
    1. Generate a numbered plan (LLM call).
    2. Execute each plan step using the ReAct loop.
    """

    AGENT_TYPE = "plan_act"

    def run(
        self,
        task: Dict[str, Any],
        strategies: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, AgentTrace]:
        task_id = task.get("id", "unknown")
        task_description = task.get("description", "")
        env = task.get("env")

        trace = AgentTrace(
            task_id=task_id,
            task_description=task_description,
        )

        # --- Phase 1: Planning ---
        plan = self._generate_plan(task, strategies or [])
        logger.info("Plan generated for task %s: %d steps", task_id, len(plan))

        # Inject plan into the conversation as context
        plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        messages = self._build_initial_messages(task, strategies or [])
        messages.append({
            "role": "assistant",
            "content": (
                f"I have created the following plan to solve this task:\n{plan_text}\n\n"
                "I will now execute each step."
            ),
        })

        context_limit = 6000
        global_step = 0

        for plan_step_idx, plan_step in enumerate(plan):
            messages.append({
                "role": "user",
                "content": f"Now execute plan step {plan_step_idx + 1}: {plan_step}",
            })

            # Execute this plan step with up to max_steps/len(plan) iterations
            steps_budget = max(3, self.max_steps // max(len(plan), 1))

            for _ in range(steps_budget):
                if global_step >= self.max_steps:
                    break
                global_step += 1

                response_text = self.llm.chat(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                thought, action_str = self._parse_response(response_text)

                if action_str.lower().startswith("action: finish") or "finish(" in action_str.lower():
                    # Step completed — move to next plan step
                    trace.steps.append(
                        TraceStep(
                            step=global_step,
                            thought=thought,
                            action=action_str,
                            observation=f"Plan step {plan_step_idx + 1} completed.",
                            success=True,
                        )
                    )
                    messages.append({"role": "assistant", "content": response_text})
                    break

                observation = self._execute_action(action_str, env)
                if len(observation) > context_limit:
                    observation = observation[:context_limit] + "\n[TRUNCATED]"

                step_success = not observation.lower().startswith("error")
                trace.steps.append(
                    TraceStep(
                        step=global_step,
                        thought=thought,
                        action=action_str,
                        observation=observation,
                        success=step_success,
                    )
                )

                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

        # Final success check
        success = self._check_task_success(task, env, "")
        trace.final_success = success
        trace.total_steps = global_step
        return success, trace

    # ------------------------------------------------------------------

    def _generate_plan(
        self,
        task: Dict[str, Any],
        strategies: List[Dict[str, Any]],
    ) -> List[str]:
        strategy_block = ""
        if strategies:
            strat_texts = "\n".join(
                f"- {s.get('strategy_text', '')}" for s in strategies
            )
            strategy_block = f"\nRelevant past strategies:\n{strat_texts}\n"

        prompt = (
            f"You are planning how to solve the following task step by step.\n"
            f"{strategy_block}\n"
            f"Task: {task.get('description', '')}\n\n"
            "Generate a numbered list of concrete steps to complete this task. "
            "Be specific and actionable. Output ONLY the numbered list, no preamble."
        )

        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.3,
            max_tokens=600,
        )

        steps = []
        for line in response.strip().splitlines():
            line = line.strip()
            # Match "1. step text" or "- step text"
            m = re.match(r"^[\d]+[.)]\s+(.+)$", line) or re.match(r"^[-*]\s+(.+)$", line)
            if m:
                steps.append(m.group(1).strip())

        return steps if steps else [task.get("description", "Complete the task.")]

    def _build_system_prompt(self, strategies: List[Dict[str, Any]]) -> str:
        base = super()._build_system_prompt(strategies)
        return base + (
            "\n\nYou are in Plan-and-Act mode. You have already generated a plan. "
            "Execute each plan step precisely, then call finish() when done."
        )
