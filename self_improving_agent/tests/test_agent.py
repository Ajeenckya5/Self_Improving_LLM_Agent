"""
Tests for agent execution using the mock LLM (no API calls).
"""

import os
import pytest

# Force mock LLM for all tests in this module
os.environ.setdefault("MOCK_LLM", "1")

from ..agent.base_agent import BaseAgent
from ..agent.plan_act_agent import PlanActAgent
from ..agent.strategy_agent import StrategyAgent


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")


def _minimal_config():
    return {
        "model": {"primary": "gpt-4", "analyzer": "gpt-4", "embedding": "all-MiniLM-L6-v2"},
        "agent": {"max_steps": 5, "temperature": 0.7, "max_tokens": 200},
        "memory": {"top_k": 3, "similarity_threshold": 0.65, "db_path": "results/test.db"},
        "logging": {"llm_calls_log": "results/test_llm_calls.jsonl"},
    }


def _simple_task():
    return {
        "id": "test_001",
        "description": "Create a file called hello.txt with content 'world'",
        "horizon": 5,
    }


class TestBaseAgent:
    def test_run_returns_tuple(self):
        agent = BaseAgent(config=_minimal_config())
        success, trace = agent.run(_simple_task())
        assert isinstance(success, bool)
        assert trace.task_id == "test_001"

    def test_trace_has_steps(self):
        agent = BaseAgent(config=_minimal_config())
        _, trace = agent.run(_simple_task())
        assert trace.total_steps >= 1

    def test_run_with_strategies(self):
        agent = BaseAgent(config=_minimal_config())
        strategies = [
            {"strategy_text": "Verify file creation before finishing.", "tags": ["file"]},
        ]
        success, trace = agent.run(_simple_task(), strategies=strategies)
        assert isinstance(success, bool)

    def test_trace_to_dict(self):
        agent = BaseAgent(config=_minimal_config())
        _, trace = agent.run(_simple_task())
        d = trace.to_dict()
        assert "task_id" in d
        assert "steps" in d
        assert isinstance(d["steps"], list)


class TestPlanActAgent:
    def test_run_returns_tuple(self):
        agent = PlanActAgent(config=_minimal_config())
        success, trace = agent.run(_simple_task())
        assert isinstance(success, bool)
        assert trace is not None

    def test_generates_plan(self):
        agent = PlanActAgent(config=_minimal_config())
        plan = agent._generate_plan(_simple_task(), [])
        assert isinstance(plan, list)
        assert len(plan) >= 1


class TestStrategyAgent:
    def test_run_without_retriever(self):
        agent = StrategyAgent(config=_minimal_config(), retriever=None)
        success, trace = agent.run(_simple_task(), strategies=[])
        assert isinstance(success, bool)

    def test_system_prompt_includes_strategies(self):
        agent = StrategyAgent(config=_minimal_config())
        strategies = [
            {"strategy_text": "Always check file permissions first.", "tags": ["file"]},
        ]
        prompt = agent._build_system_prompt(strategies)
        assert "Always check file permissions first." in prompt


class TestActionParsing:
    def test_parse_valid_response(self):
        agent = BaseAgent(config=_minimal_config())
        text = "Thought: I should check the file.\nAction: bash(ls -la)"
        thought, action = agent._parse_response(text)
        assert "check the file" in thought
        assert "bash" in action

    def test_parse_finish_action(self):
        agent = BaseAgent(config=_minimal_config())
        text = "Thought: Done.\nAction: finish(Task complete)"
        thought, action = agent._parse_response(text)
        assert "finish" in action.lower()
