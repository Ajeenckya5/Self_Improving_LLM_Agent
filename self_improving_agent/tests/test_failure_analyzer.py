"""
Tests for FailureAnalyzer — uses canned traces, no API calls needed.
"""

import pytest
from ..analysis.failure_analyzer import FailureAnalyzer
from ..agent.base_agent import AgentTrace, TraceStep


def _make_trace(steps_data):
    """Build an AgentTrace from list of (thought, action, observation, success)."""
    trace = AgentTrace(task_id="test", task_description="test task")
    for i, (thought, action, obs, success) in enumerate(steps_data, start=1):
        trace.steps.append(TraceStep(step=i, thought=thought, action=action, observation=obs, success=success))
    trace.total_steps = len(steps_data)
    return trace


@pytest.fixture
def analyzer():
    return FailureAnalyzer()


# ---------------------------------------------------------------------------
# Repeated action detection
# ---------------------------------------------------------------------------

class TestRepeatedAction:
    def test_detects_three_identical_actions(self, analyzer):
        trace = _make_trace([
            ("I'll try ls", "Action: bash(ls -la)", "total 0", True),
            ("Let me try again", "Action: bash(ls -la)", "total 0", False),
            ("One more time", "Action: bash(ls -la)", "total 0", False),
        ])
        result = analyzer.analyze(trace)
        assert result["failure_type"] == "repeated_action"
        assert len(result["failed_steps"]) >= 1

    def test_no_false_positive_with_two_identical(self, analyzer):
        trace = _make_trace([
            ("Try ls", "Action: bash(ls)", "file.txt", True),
            ("Try ls once more", "Action: bash(ls)", "file.txt", True),
            ("Now read", "Action: read_file(file.txt)", "content", True),
        ])
        result = analyzer.analyze(trace)
        # Should NOT flag repeated_action (only 2 occurrences)
        assert result["failure_type"] != "repeated_action"


# ---------------------------------------------------------------------------
# Circular loop detection
# ---------------------------------------------------------------------------

class TestCircularLoop:
    def test_detects_two_step_loop(self, analyzer):
        trace = _make_trace([
            ("Check dir", "Action: bash(pwd)", "/home/user", True),
            ("List files", "Action: bash(ls)", "a.txt", True),
            ("Check dir again", "Action: bash(pwd)", "/home/user", True),
            ("List files again", "Action: bash(ls)", "a.txt", False),
        ])
        result = analyzer.analyze(trace)
        # Either circular_loop or repeated_action — both indicate the loop
        assert result["failure_type"] in ("circular_loop", "repeated_action")

    def test_no_loop_in_linear_trace(self, analyzer):
        trace = _make_trace([
            ("Step 1", "Action: bash(mkdir test)", "", True),
            ("Step 2", "Action: bash(cd test)", "", True),
            ("Step 3", "Action: write_file(a.txt, hello)", "Written", True),
            ("Done", "Action: finish(done)", "Task declared complete.", True),
        ])
        result = analyzer.analyze(trace)
        assert result["failure_type"] not in ("circular_loop",)


# ---------------------------------------------------------------------------
# Context truncation detection
# ---------------------------------------------------------------------------

class TestContextTruncation:
    def test_detects_truncation_marker(self, analyzer):
        long_obs = "x" * 100 + "\n[TRUNCATED: observation exceeded context limit]"
        trace = _make_trace([
            ("Run big command", "Action: bash(cat bigfile.txt)", long_obs, False),
            ("Try again", "Action: bash(cat bigfile.txt)", long_obs, False),
        ])
        result = analyzer.analyze(trace)
        assert result["failure_type"] == "context_truncation"

    def test_detects_very_long_observation(self, analyzer):
        # 4096 * 4 = 16384 chars ~ 4096 tokens → exceeds 80% of 4096 token window
        very_long = "data " * 4000
        trace = _make_trace([
            ("Read huge file", "Action: bash(cat /dev/urandom)", very_long, False),
        ])
        result = analyzer.analyze(trace)
        assert result["failure_type"] == "context_truncation"


# ---------------------------------------------------------------------------
# Tool misuse detection
# ---------------------------------------------------------------------------

class TestToolMisuse:
    def test_detects_parse_error(self, analyzer):
        trace = _make_trace([
            ("Try something", "bash ls -la", "Error: Could not parse action. Use the format: Action: tool_name(arguments)", False),
            ("Try again", "bash(ls", "Error: Could not parse action.", False),
            ("One more", "Action:bash(ls)", "Error: Could not parse action.", False),
        ])
        result = analyzer.analyze(trace)
        assert result["failure_type"] in ("tool_misuse", "repeated_action")

    def test_no_misuse_on_clean_errors(self, analyzer):
        trace = _make_trace([
            ("Check file", "Action: bash(cat missing.txt)", "cat: missing.txt: No such file or directory", False),
            ("Create file", "Action: write_file(missing.txt, hello)", "Written 5 bytes", True),
            ("Done", "Action: finish(done)", "", True),
        ])
        result = analyzer.analyze(trace)
        # Single tool error followed by success — should NOT be tool_misuse
        assert result["failure_type"] != "tool_misuse"


# ---------------------------------------------------------------------------
# Dict-based trace input
# ---------------------------------------------------------------------------

class TestDictTrace:
    def test_handles_dict_input(self, analyzer):
        trace = {
            "task_id": "t1",
            "task_description": "test",
            "steps": [
                {"step": 1, "thought": "try", "action": "Action: bash(ls)", "observation": "ok", "success": True},
                {"step": 2, "thought": "try again", "action": "Action: bash(ls)", "observation": "ok", "success": True},
                {"step": 3, "thought": "try again", "action": "Action: bash(ls)", "observation": "ok", "success": True},
            ],
        }
        result = analyzer.analyze(trace)
        assert result["failure_type"] == "repeated_action"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_trace(self, analyzer):
        trace = AgentTrace(task_id="t", task_description="empty")
        result = analyzer.analyze(trace)
        assert "failure_type" in result

    def test_single_step(self, analyzer):
        trace = _make_trace([
            ("Only step", "Action: bash(ls)", "Error: permission denied", False),
        ])
        result = analyzer.analyze(trace)
        assert "failure_type" in result
        assert "pattern_summary" in result
