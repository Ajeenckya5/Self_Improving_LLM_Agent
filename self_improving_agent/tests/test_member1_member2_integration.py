"""Integration tests for Member 1 (trace collection) and Member 2 (failure analysis)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from self_improving_agent.analysis.failure_analyzer import FailureAnalyzer
from self_improving_agent.analysis.llm_failure_analyzer import LLMFailureAnalyzer
from self_improving_agent.utils.trace_logger import ExecutionTrace, TraceLogger, TraceStep


class DummyLLMForIntegration:
    """Dummy LLM that returns consistent failure analysis results."""

    def chat(self, messages, model="gpt-4", temperature=0.0, max_tokens=1000):
        prompt_text = messages[-1]["content"].lower() if messages else ""

        # Return failure analysis based on the heuristic hint in the prompt
        if "heuristic" in prompt_text:
            if "repeated" in prompt_text:
                return json.dumps({
                    "failure_type": "repeated_action",
                    "failed_steps": [2, 3, 4],
                    "pattern_summary": "The agent repeatedly executed the same command.",
                    "confidence": 0.95,
                })
            elif "tool" in prompt_text or "misuse" in prompt_text:
                return json.dumps({
                    "failure_type": "tool_misuse",
                    "failed_steps": [1, 2],
                    "pattern_summary": "The agent used incorrect syntax.",
                    "confidence": 0.9,
                })

        return json.dumps({
            "failure_type": "other",
            "failed_steps": [],
            "pattern_summary": "Unable to determine failure type.",
            "confidence": 0.5,
        })


def _load_member2_eval_data() -> List[Dict[str, Any]]:
    """Load the member2 evaluation traces."""
    data_path = Path("self_improving_agent/data/member2_eval_traces.jsonl")
    traces = []
    if data_path.exists():
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(json.loads(line))
    return traces


def trace_dict_to_execution_trace(trace_dict: Dict[str, Any]) -> ExecutionTrace:
    """Convert a trace dict (from JSONL) to an ExecutionTrace object (Member 1's format)."""
    exec_trace = ExecutionTrace(
        task_id=trace_dict.get("id", "unknown"),
        agent_type="plan_act",
        attempt=1,
    )

    for step_dict in trace_dict.get("steps", []):
        trace_step = TraceStep(
            step=step_dict["step"],
            state={},
            action={"action_str": step_dict["action"]},
            observation=step_dict["observation"],
            reasoning=step_dict.get("thought", ""),
        )
        exec_trace.steps.append(trace_step)

    exec_trace.success = all(step.get("success", False) for step in trace_dict.get("steps", []))
    return exec_trace


# ============================================================================
# Integration Tests: Member 1 → Member 2 pipeline
# ============================================================================


class TestMember1Member2Integration:
    """Test the integration of Member 1 trace collection and Member 2 failure analysis."""

    def test_member1_trace_logger_with_member2_analyzer_repeated_action(self):
        """
        Integration test: Member 1 creates a trace with repeated actions,
        Member 2 analyzes it and detects the failure pattern.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Member 1: Create a trace using TraceLogger
            logger = TraceLogger(traces_dir=tmpdir)
            logger.start_trace(task_id="test_repeated_action", agent_type="plan_act", attempt=1)

            # Simulate repeated action failure
            for i in range(4):
                logger.log_step(
                    state={"step": i + 1},
                    action={"cmd": "ls workspace"},
                    observation="data.csv notes.md src/",
                    reasoning=f"Attempt {i + 1} to find file",
                )

            trace = logger.end_trace(success=False, message="Task failed")
            assert trace is not None
            assert len(trace.steps) == 4

            # Member 2: Analyze the trace from Member 1
            analyzer = FailureAnalyzer()
            result = analyzer.analyze(trace)

            # Verify Member 2 correctly identified the repeated action
            assert result["failure_type"] == "repeated_action"
            assert len(result["failed_steps"]) == 4
            assert "repeated" in result["pattern_summary"].lower()

    def test_member1_member2_tool_misuse_detection(self):
        """
        Integration test: Tool misuse errors are correctly detected.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TraceLogger(traces_dir=tmpdir)
            logger.start_trace(task_id="test_tool_misuse", agent_type="plan_act", attempt=1)

            # Simulate tool misuse with clear error signals
            logger.log_step(
                state={},
                action={"cmd": "query_sql(SELECT ...)"},
                observation="Error: could not parse command - SQL syntax error",
                reasoning="Query the database",
            )
            logger.log_step(
                state={},
                action={"cmd": "query_sql(SELECT ...)"},
                observation="Error: tool argument format invalid - unexpected token",
                reasoning="Try with different syntax",
            )

            trace = logger.end_trace(success=False)

            # Analyze with Member 2
            analyzer = FailureAnalyzer()
            result = analyzer.analyze(trace)

            # Tool misuse should be detected or at least not crash
            assert result["failure_type"] in ["tool_misuse", "other"]
            assert "failed_steps" in result

    def test_member1_member2_planning_error_detection(self):
        """
        Integration test: Planning errors (wrong sequence) are detected or handled gracefully.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TraceLogger(traces_dir=tmpdir)
            logger.start_trace(task_id="test_planning_error", agent_type="plan_act", attempt=1)

            # Simulate planning error (wrong sequence of operations)
            logger.log_step(
                state={},
                action={"cmd": "navigate(/profile)"},
                observation="Error: authentication required; redirecting to /login.",
                reasoning="Go to profile page",
            )
            logger.log_step(
                state={},
                action={"cmd": "type(#bio, text)"},
                observation="Error: selector #bio not found on /login.",
                reasoning="Fill bio field",
            )

            trace = logger.end_trace(success=False)

            # Analyze with Member 2
            analyzer = FailureAnalyzer()
            result = analyzer.analyze(trace)

            # The analyzer should handle this without crashing
            # It may detect incorrect_reasoning, planning_error, or other
            assert "failure_type" in result
            assert "failed_steps" in result
            # At minimum, it should identify something failed
            assert len(result["failed_steps"]) > 0 or result["failure_type"] == "other"

    def test_member1_save_load_member2_analyze(self):
        """
        Integration test: Full pipeline - Member 1 saves trace to disk,
        then loads and Member 2 analyzes it.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Member 1: Create, log, and save
            logger = TraceLogger(traces_dir=tmpdir)
            logger.start_trace(task_id="fs_repeat_001", agent_type="plan_act", attempt=1)

            for i in range(3):
                logger.log_step(
                    state={"workspace_state": "unchanged"},
                    action={"cmd": "ls workspace"},
                    observation="data.csv notes.md src/",
                    reasoning=f"List iteration {i + 1}",
                )

            trace = logger.end_trace(success=False, message="File not found")
            saved_path = logger.save_trace(trace, run_id="integration_test")

            assert saved_path.exists()

            # Member 1: Reload from disk
            reloaded_trace = logger.load_trace(saved_path)
            assert reloaded_trace.task_id == "fs_repeat_001"
            assert len(reloaded_trace.steps) == 3

            # Member 2: Analyze the loaded trace
            analyzer = FailureAnalyzer()
            result = analyzer.analyze(reloaded_trace)

            assert result["failure_type"] == "repeated_action"
            assert result["failed_steps"] == [1, 2, 3]

    def test_member2_llm_analyzer_with_member1_trace(self):
        """
        Integration test: Member 2's LLM analyzer works with Member 1's traces.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Member 1: Create a trace
            logger = TraceLogger(traces_dir=tmpdir)
            logger.start_trace(task_id="web_plan_001", agent_type="plan_act", attempt=1)

            logger.log_step(
                state={},
                action={"cmd": "navigate(/profile)"},
                observation="Error: authentication required.",
                reasoning="Navigate to profile",
            )
            logger.log_step(
                state={},
                action={"cmd": "type(#bio, Research)"},
                observation="Error: selector not found.",
                reasoning="Fill bio",
            )

            trace = logger.end_trace(success=False)

            # Member 2: Use LLM analyzer with the trace
            config = {
                "model": {"analyzer": "gpt-4"},
                "analysis": {"failure_prompt": "failure_analysis.txt"},
            }
            llm_analyzer = LLMFailureAnalyzer(
                config=config,
                llm_client=DummyLLMForIntegration(),
                fallback=FailureAnalyzer(),
            )

            task = {"id": "web_plan_001", "description": "Create account and fill profile"}
            result = llm_analyzer.analyze(task, trace)

            # Verify result structure
            assert "failure_type" in result
            assert "failed_steps" in result
            assert result["analysis_source"] in ["llm", "heuristic_fallback"]

    def test_member1_member2_context_truncation(self):
        """
        Integration test: Very long observations are detected as context truncation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TraceLogger(traces_dir=tmpdir)
            logger.start_trace(task_id="test_context_truncation", agent_type="plan_act", attempt=1)

            # Log step with very long observation
            long_observation = "x" * 50000
            logger.log_step(
                state={},
                action={"cmd": "ls -la"},
                observation=long_observation,
                reasoning="List files",
            )

            trace = logger.end_trace(success=False)

            # Analyze
            analyzer = FailureAnalyzer()
            result = analyzer.analyze(trace)

            # Should detect context truncation
            assert result["failure_type"] == "context_truncation"

    def test_integration_with_real_eval_data(self):
        """
        Integration test with real data from member2_eval_traces.jsonl.
        Tests Member 1 and Member 2 on realistic failure scenarios.
        """
        traces_data = _load_member2_eval_data()
        if not traces_data:
            pytest.skip("member2_eval_traces.jsonl not found")

        analyzer = FailureAnalyzer()

        for trace_dict in traces_data:
            # Convert dict trace to ExecutionTrace (Member 1's format)
            exec_trace = trace_dict_to_execution_trace(trace_dict)

            # Analyze with Member 2
            result = analyzer.analyze(exec_trace)

            # Verify result structure
            assert "failure_type" in result
            assert "failed_steps" in result
            assert "pattern_summary" in result

            # Check detected type
            gold_type = trace_dict.get("gold_failure_type", "other")
            detected_type = result["failure_type"]

            print(f"\nTask: {trace_dict.get('id')}")
            print(f"  Gold type: {gold_type}")
            print(f"  Detected type: {detected_type}")
            print(f"  Pattern: {result['pattern_summary']}")

    def test_member1_member2_multiple_failure_types(self):
        """
        Integration test: Verify Member 2 can distinguish between different failure types
        produced by Member 1.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TraceLogger(traces_dir=tmpdir)
            analyzer = FailureAnalyzer()

            test_cases = [
                {
                    "task_id": "repeated",
                    "steps": [
                        ("Attempt 1", {"cmd": "cmd1"}, "result1"),
                        ("Attempt 2", {"cmd": "cmd1"}, "result1"),
                        ("Attempt 3", {"cmd": "cmd1"}, "result1"),
                    ],
                    "expected_type": "repeated_action",
                },
                {
                    "task_id": "tool_error",
                    "steps": [
                        ("Try query", {"cmd": "query(bad)"}, "Error: syntax error"),
                        ("Try again", {"cmd": "query(bad)"}, "Error: syntax error"),
                    ],
                    "expected_type": "tool_misuse",
                },
            ]

            results_by_type = {}

            for test_case in test_cases:
                logger.start_trace(
                    task_id=test_case["task_id"],
                    agent_type="plan_act",
                    attempt=1,
                )

                for thought, action, obs in test_case["steps"]:
                    logger.log_step(
                        state={},
                        action=action,
                        observation=obs,
                        reasoning=thought,
                    )

                trace = logger.end_trace(success=False)
                result = analyzer.analyze(trace)
                results_by_type[test_case["task_id"]] = result["failure_type"]

            # Verify different failure types were detected
            failure_types = list(results_by_type.values())
            assert len(set(failure_types)) > 1, "Should detect different failure types"


class TestMember1TraceFormat:
    """Test that Member 1's trace format works with Member 2."""

    def test_execution_trace_has_required_fields(self):
        """Verify ExecutionTrace has all fields that FailureAnalyzer needs."""
        trace = ExecutionTrace(task_id="test", agent_type="plan_act", attempt=1)
        trace.steps.append(
            TraceStep(
                step=1,
                state={},
                action={},
                observation="test obs",
                reasoning="test reasoning",
            )
        )

        analyzer = FailureAnalyzer()
        steps = analyzer._extract_steps(trace)

        assert len(steps) == 1
        assert steps[0]["step"] == 1
        assert steps[0]["observation"] == "test obs"
        # Should use reasoning as thought for Member 1's TraceStep
        assert steps[0]["thought"] == "test reasoning"

    def test_trace_step_fields_accessible(self):
        """Verify TraceStep fields are properly accessible."""
        step = TraceStep(
            step=1,
            state={"env": "test"},
            action={"cmd": "test_cmd"},
            observation="test result",
            reasoning="test reasoning",
        )

        assert step.step == 1
        assert step.observation == "test result"
        assert step.reasoning == "test reasoning"
        assert step.action == {"cmd": "test_cmd"}


class TestMember2AnalysisAccuracy:
    """Test that Member 2 analysis is accurate on Member 1 traces."""

    def test_repeated_action_detection_accuracy(self):
        """Test accuracy of repeated action detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TraceLogger(traces_dir=tmpdir)
            logger.start_trace(task_id="repeat_test", agent_type="plan_act", attempt=1)

            # Log same action 5 times
            for i in range(5):
                logger.log_step(
                    state={},
                    action={"cmd": "list_files"},
                    observation=f"files: {i}",
                    reasoning=f"Attempt {i}",
                )

            trace = logger.end_trace(success=False)
            analyzer = FailureAnalyzer()
            result = analyzer.analyze(trace)

            assert result["failure_type"] == "repeated_action"
            assert len(result["failed_steps"]) == 5

    def test_tool_misuse_detection_accuracy(self):
        """Test accuracy of tool misuse detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TraceLogger(traces_dir=tmpdir)
            logger.start_trace(task_id="tool_test", agent_type="plan_act", attempt=1)

            logger.log_step(
                state={},
                action={"cmd": "execute_query"},
                observation="Error: could not parse command",
                reasoning="Execute query",
            )
            logger.log_step(
                state={},
                action={"cmd": "execute_query"},
                observation="Error: tool argument format invalid",
                reasoning="Try again",
            )

            trace = logger.end_trace(success=False)
            analyzer = FailureAnalyzer()
            result = analyzer.analyze(trace)

            assert result["failure_type"] == "tool_misuse"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
