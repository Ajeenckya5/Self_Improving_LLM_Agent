"""Tests for recurrence_analysis module."""

from __future__ import annotations

import pandas as pd
import pytest

from ..analysis.recurrence_analysis import (
    compare_recurrence_across_conditions,
    failure_recurrence_over_time,
)


def _make_df(successes: list[bool], failure_types: list[str | None]) -> pd.DataFrame:
    return pd.DataFrame({"success": successes, "failure_type": failure_types})


class TestFailureRecurrenceOverTime:
    def test_empty_on_all_success(self):
        df = _make_df([True, True, True], [None, None, None])
        result = failure_recurrence_over_time(df)
        assert result.empty

    def test_no_recurrence_unique_failures(self):
        df = _make_df(
            [False, False, False],
            ["repeated_action", "circular_loop", "tool_misuse"],
        )
        result = failure_recurrence_over_time(df)
        # First failure can't recur; subsequent ones are unique — rate stays 0
        assert result["recurrence_rate"].iloc[-1] == 0.0

    def test_full_recurrence(self):
        df = _make_df(
            [False, False, False, False],
            ["repeated_action", "repeated_action", "repeated_action", "repeated_action"],
        )
        result = failure_recurrence_over_time(df)
        # After first, all subsequent are recurring
        assert result["recurrence_rate"].iloc[-1] > 0.5

    def test_returns_dataframe_with_correct_columns(self):
        df = _make_df([False, False], ["tool_misuse", "tool_misuse"])
        result = failure_recurrence_over_time(df)
        assert set(result.columns) == {"task_n", "recurrence_rate"}


class TestCompareRecurrenceAcrossConditions:
    def test_returns_one_row_per_condition(self):
        results = {
            "ReAct": _make_df([False, False], ["repeated_action", "repeated_action"]),
            "Self-Improving": _make_df([True, True], [None, None]),
        }
        table = compare_recurrence_across_conditions(results)
        assert len(table) == 2

    def test_high_recurrence_for_same_failure_type(self):
        results = {
            "Baseline": _make_df(
                [False, False, False],
                ["repeated_action", "repeated_action", "repeated_action"],
            )
        }
        table = compare_recurrence_across_conditions(results)
        assert table.iloc[0]["recurrence_rate"] > 0.0

    def test_zero_recurrence_all_unique(self):
        results = {
            "Good": _make_df(
                [False, False, False],
                ["repeated_action", "circular_loop", "tool_misuse"],
            )
        }
        table = compare_recurrence_across_conditions(results)
        assert table.iloc[0]["recurrence_rate"] == 0.0

    def test_all_success_no_failures(self):
        results = {"Perfect": _make_df([True, True], [None, None])}
        table = compare_recurrence_across_conditions(results)
        assert table.iloc[0]["n_failures"] == 0
        assert table.iloc[0]["recurrence_rate"] == 0.0
