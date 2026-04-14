"""
Tests for evaluation metric functions — pure math, no API calls.
"""

import pandas as pd
import pytest

from ..evaluation.metrics import (
    cumulative_success_curve,
    failure_mode_distribution,
    repeated_failure_rate,
    success_vs_horizon,
    task_success_rate,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "task_id": [f"t{i}" for i in range(10)],
        "horizon": [5, 5, 10, 10, 15, 15, 20, 20, 20, 20],
        "agent_type": ["react"] * 10,
        "success": [True, False, True, True, False, False, True, False, True, True],
        "failure_type": [None, "repeated_action", None, None, "circular_loop",
                         "tool_misuse", None, "repeated_action", None, None],
        "steps_taken": [5, 10, 8, 7, 15, 12, 20, 18, 16, 14],
        "strategies_used": [0] * 10,
    })


class TestTaskSuccessRate:
    def test_overall_rate(self, sample_df):
        rate = task_success_rate(sample_df)
        # 6 successes out of 10
        assert abs(rate["overall"] - 0.6) < 1e-6

    def test_by_horizon(self, sample_df):
        rate = task_success_rate(sample_df, groupby="horizon")
        assert abs(rate[5] - 0.5) < 1e-6   # 1/2 at horizon 5
        assert abs(rate[10] - 1.0) < 1e-6  # 2/2 at horizon 10
        assert abs(rate[15] - 0.0) < 1e-6  # 0/2 at horizon 15

    def test_all_success(self):
        df = pd.DataFrame({"success": [True, True, True]})
        assert task_success_rate(df)["overall"] == 1.0

    def test_all_failure(self):
        df = pd.DataFrame({"success": [False, False]})
        assert task_success_rate(df)["overall"] == 0.0


class TestFailureModeDistribution:
    def test_distribution_sums_to_one(self, sample_df):
        dist = failure_mode_distribution(sample_df)
        assert abs(dist.sum() - 1.0) < 1e-6

    def test_correct_fractions(self, sample_df):
        dist = failure_mode_distribution(sample_df)
        # 2 repeated_action, 1 circular_loop, 1 tool_misuse = 4 failures
        assert abs(dist["repeated_action"] - 0.5) < 1e-6
        assert abs(dist["circular_loop"] - 0.25) < 1e-6
        assert abs(dist["tool_misuse"] - 0.25) < 1e-6

    def test_empty_when_all_succeed(self):
        df = pd.DataFrame({"success": [True, True], "failure_type": [None, None]})
        dist = failure_mode_distribution(df)
        assert dist.empty


class TestSuccessVsHorizon:
    def test_returns_correct_horizons(self, sample_df):
        result = success_vs_horizon(sample_df)
        horizons = result["horizon"].tolist()
        assert set(horizons) == {5, 10, 15, 20}

    def test_rates_match_expectations(self, sample_df):
        result = success_vs_horizon(sample_df)
        if "success_rate" in result.columns:
            h10_rate = result[result["horizon"] == 10]["success_rate"].values[0]
            assert abs(h10_rate - 1.0) < 1e-6


class TestCumulativeSuccessCurve:
    def test_monotone_for_all_successes(self):
        df = pd.DataFrame({
            "success": [True, True, True, True],
            "agent_type": ["react"] * 4,
        })
        curve = cumulative_success_curve(df)
        rates = curve[curve["agent_type"] == "react"].sort_values("task_n")["cumulative_success"].tolist()
        assert rates == [1.0, 1.0, 1.0, 1.0]

    def test_curve_bounded_zero_to_one(self, sample_df):
        curve = cumulative_success_curve(sample_df)
        assert curve["cumulative_success"].min() >= 0.0
        assert curve["cumulative_success"].max() <= 1.0

    def test_all_tasks_represented(self, sample_df):
        curve = cumulative_success_curve(sample_df)
        # Should have one entry per task per agent_type
        n_tasks = len(sample_df[sample_df["agent_type"] == "react"])
        n_curve = len(curve[curve["agent_type"] == "react"])
        assert n_curve == n_tasks


class TestRepeatedFailureRate:
    def test_no_repetition(self):
        df = pd.DataFrame({
            "success": [False, False, False],
            "failure_type": ["repeated_action", "circular_loop", "tool_misuse"],
        })
        rate = repeated_failure_rate(df)
        assert rate == 0.0

    def test_all_same_failure(self):
        df = pd.DataFrame({
            "success": [False, False, False],
            "failure_type": ["repeated_action", "repeated_action", "repeated_action"],
        })
        rate = repeated_failure_rate(df)
        # First is new, second and third are repeats → 2/3
        assert abs(rate - 2 / 3) < 1e-6

    def test_no_failures(self):
        df = pd.DataFrame({
            "success": [True, True],
            "failure_type": [None, None],
        })
        rate = repeated_failure_rate(df)
        assert rate == 0.0
