"""Tests for Member 2 LLM prompts, judge harness, and ablation plumbing."""

import json
from pathlib import Path

from ..analysis.llm_failure_analyzer import LLMFailureAnalyzer
from ..analysis.strategy_generator import StrategyGenerator
from ..evaluation.llm_judge import FailureStrategyJudge
from ..experiments.member2_eval import run_member2_eval


class DummyLLM:
    def chat(self, messages, model="gpt-4", temperature=0.0, max_tokens=1000):
        prompt = messages[-1]["content"].lower()
        if "evaluation judge" in prompt:
            return json.dumps({
                "failure_type_correct": True,
                "failed_steps_overlap": 1.0,
                "analysis_grounding_score": 5,
                "strategy_specificity_score": 4,
                "strategy_actionability_score": 5,
                "retrieval_tags_score": 4,
                "overall_score": 5,
                "rationale": "The answer is grounded and actionable.",
            })
        if "corrective strategy" in prompt or "retrieval-ready" in prompt:
            return json.dumps({
                "strategy_text": "The agent skipped account creation before editing the profile. Navigate to signup, create the account, wait for the profile page, type the bio, save it, and verify the saved text appears.",
                "decision_rule": "Always complete authentication before using profile selectors.",
                "tags": ["planning_error", "signup", "profile_state"],
            })
        return json.dumps({
            "failure_type": "planning_error",
            "failed_steps": [1, 2],
            "pattern_summary": "The agent attempted profile edits before creating an account. The trace shows authentication errors and missing selectors.",
            "confidence": 0.9,
        })


def _config():
    return {
        "model": {
            "analyzer": "grok-4.20-reasoning",
            "judge": "grok-4.20-reasoning",
            "backend": "mock",
        },
        "analysis": {
            "failure_prompt": "failure_analysis.txt",
            "strategy_prompt": "strategy_gen.txt",
            "judge_prompt": "evaluation_judge.txt",
            "judge_model": "grok-4.20-reasoning",
        },
        "logging": {"llm_calls_log": "results/test_llm_calls.jsonl"},
    }


def _task():
    return {
        "id": "web_plan_001",
        "description": "Create an account, fill the profile bio, and save it.",
        "gold_failure_type": "planning_error",
        "gold_failed_steps": [1, 2],
    }


def _trace():
    return {
        "steps": [
            {
                "step": 1,
                "thought": "Go to profile.",
                "action": "Action: navigate(/profile)",
                "observation": "Error: authentication required.",
                "success": False,
            },
            {
                "step": 2,
                "thought": "Type the bio.",
                "action": "Action: type(#bio, hello)",
                "observation": "Error: selector #bio not found.",
                "success": False,
            },
        ]
    }


def test_llm_failure_analyzer_validates_json():
    analyzer = LLMFailureAnalyzer(config=_config(), llm_client=DummyLLM())
    result = analyzer.analyze(_task(), _trace())
    assert result["failure_type"] == "planning_error"
    assert result["failed_steps"] == [1, 2]
    assert result["analysis_source"] == "llm"


def test_strategy_generator_includes_decision_rule():
    generator = StrategyGenerator(config=_config(), llm_client=DummyLLM())
    strategy = generator.generate(
        _task(),
        {
            "failure_type": "planning_error",
            "failed_steps": [1, 2],
            "pattern_summary": "Profile actions happened before signup.",
            "raw_trace_excerpt": "Step 1 navigate profile -> auth required",
        },
    )
    assert "Decision rule:" in strategy["strategy_text"]
    assert "planning_error" in strategy["tags"]


def test_failure_strategy_judge_scores_candidate():
    judge = FailureStrategyJudge(config=_config(), llm_client=DummyLLM())
    scores = judge.evaluate(
        _task(),
        _trace(),
        {"failure_type": "planning_error", "failed_steps": [1, 2]},
        {"strategy_text": "Create account before profile edits.", "tags": ["planning_error"]},
    )
    assert scores["failure_type_correct"] is True
    assert scores["overall_score"] == 5


def test_member2_eval_harness_writes_csv(tmp_path):
    dataset = Path("self_improving_agent/data/member2_eval_traces.jsonl")
    output = tmp_path / "member2_eval.csv"
    df = run_member2_eval(
        dataset_path=dataset,
        output_path=output,
        config=_config(),
        analyzer_mode="llm",
    )
    assert output.exists()
    assert output.with_suffix(".jsonl").exists()
    assert len(df) >= 1
