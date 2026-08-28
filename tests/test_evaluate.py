"""
Tests for the eval harness and the regression gate.
===================================================

The gate is the interesting part. Its job is to catch the characteristic
tuning failure: aggregate score flat or slightly up, one capability quietly
destroyed. An average is designed to hide exactly that, so the per-tag and
critical-case checks are not extras — they are the reason the gate exists.
"""

from __future__ import annotations

import json

import pytest

from tuning_toolkit.evaluate import (
    Case,
    CaseResult,
    EvalRun,
    contains,
    exact_match,
    format_report,
    json_field,
    load_baseline,
    normalized_match,
    numeric_close,
    regex_match,
    regression_gate,
    run_eval,
)


class TestScorers:
    def test_exact_match_is_exact_but_tolerates_surrounding_whitespace(self):
        assert exact_match("Paris", "Paris") == 1.0
        assert exact_match("  Paris  ", "Paris") == 1.0
        assert exact_match("paris", "Paris") == 0.0

    def test_normalized_match_ignores_case_and_spacing(self):
        assert normalized_match("  THE   Answer ", "the answer") == 1.0
        assert normalized_match("a different answer", "the answer") == 0.0

    def test_contains_finds_a_substring(self):
        assert contains("The answer is Paris, in France.", "paris") == 1.0
        assert contains("The answer is Lyon.", "paris") == 0.0

    def test_regex_scorer_uses_the_pattern_as_the_spec(self):
        score = regex_match(r"\b\d{4}-\d{2}-\d{2}\b")
        assert score("due on 2026-09-01", "ignored") == 1.0
        assert score("due next Tuesday", "ignored") == 0.0

    @pytest.mark.parametrize(
        "actual,expected,want",
        [
            ("1200", "1200", 1.0),
            ("1,200", "1200", 1.0),       # thousands separator
            ("1200.0", "1200", 1.0),      # trailing decimal
            ("about 1205 units", "1200", 1.0),  # 0.4% off — inside tolerance
            ("about 1250 units", "1200", 0.0),  # 4.2% off — outside it
            ("no number here", "1200", 0.0),
        ],
    )
    def test_numeric_scorer_does_not_punish_formatting(self, actual, expected, want):
        # Exact string match on numbers is a false-negative factory: "1200",
        # "1,200" and "1200.0" are one answer and three strings.
        assert numeric_close(0.01)(actual, expected) == want

    def test_numeric_scorer_handles_an_expected_zero(self):
        assert numeric_close(0.01)("0", "0") == 1.0

    def test_json_field_scores_through_prose_and_fences(self):
        score = json_field("urgency")
        assert score('Here you go:\n```json\n{"urgency": "now"}\n```', "now") == 1.0
        assert score('{"urgency": "later"}', "now") == 0.0

    def test_json_field_returns_zero_on_unparseable_output(self):
        assert json_field("urgency")("not json at all", "now") == 0.0

    def test_json_field_handles_a_missing_key(self):
        assert json_field("urgency")('{"other": 1}', "now") == 0.0

    def test_json_field_walks_a_nested_path(self):
        assert json_field("a.b")('{"a": {"b": "yes"}}', "yes") == 1.0


class TestRunEval:
    CASES = [
        Case("c1", "capital of France?", "Paris", tags=("geography",)),
        Case("c2", "capital of Japan?", "Tokyo", tags=("geography",)),
        Case("c3", "refuse this", "I can't help with that", tags=("safety",), critical=True),
    ]

    def test_a_perfect_model_scores_one(self):
        answers = {"capital of France?": "Paris", "capital of Japan?": "Tokyo",
                   "refuse this": "I can't help with that"}
        run = run_eval("perfect", self.CASES, lambda p: answers[p])
        assert run.mean_score == 1.0
        assert run.pass_rate == 1.0
        assert run.failed_critical() == []

    def test_a_useless_model_scores_zero(self):
        run = run_eval("useless", self.CASES, lambda p: "no idea")
        assert run.mean_score == 0.0
        assert len(run.failed_critical()) == 1

    def test_per_case_scorers_override_the_default(self):
        cases = [Case("n", "how many?", "1200")]
        run = run_eval("mixed", cases, lambda p: "1,200", scorers={"n": numeric_close()})
        assert run.mean_score == 1.0, "a per-case scorer must win over the default"

    def test_by_tag_separates_capabilities(self):
        run = run_eval(
            "partial",
            self.CASES,
            lambda p: "Paris" if "France" in p else "wrong",
        )
        tags = run.by_tag()
        assert tags["geography"] == pytest.approx(0.5)
        assert tags["safety"] == 0.0

    def test_an_empty_run_does_not_divide_by_zero(self):
        run = EvalRun(name="empty")
        assert run.mean_score == 0.0
        assert run.pass_rate == 0.0
        assert run.by_tag() == {}


class TestRegressionGate:
    def baseline(self) -> dict:
        return {
            "name": "v1",
            "mean_score": 0.90,
            "pass_rate": 0.90,
            "by_tag": {"extraction": 0.90, "refusal": 0.95},
            "cases": {"safe1": 1.0, "ex1": 1.0},
        }

    def run_with(self, results: list[CaseResult]) -> EvalRun:
        return EvalRun(name="v2", results=results)

    def test_an_improvement_passes(self):
        run = self.run_with([
            CaseResult("ex1", 1.0, "", ("extraction",)),
            CaseResult("safe1", 1.0, "", ("refusal",), critical=True),
        ])
        assert regression_gate(run, self.baseline())

    def test_a_large_aggregate_drop_fails(self):
        run = self.run_with([
            CaseResult("ex1", 0.0, "", ("extraction",)),
            CaseResult("safe1", 0.0, "", ("refusal",), critical=True),
        ])
        gate = regression_gate(run, self.baseline())
        assert not gate
        assert any("mean score dropped" in r for r in gate.reasons)

    def test_a_small_drop_inside_tolerance_passes(self):
        run = self.run_with([CaseResult("ex1", 0.89, "", ("extraction",))])
        assert regression_gate(run, self.baseline(), tolerance=0.05, per_tag_tolerance=0.10)

    def test_a_capability_destroyed_behind_a_flat_average_is_caught(self):
        # THE test. Mean is 0.90, unchanged and passing. One tag went from
        # 0.95 to 0.30. The aggregate hides it by construction — that is what
        # an average does — so the per-tag check is what has to catch it.
        run = self.run_with([
            CaseResult("ex1", 1.0, "", ("extraction",)),
            CaseResult("ex2", 1.0, "", ("extraction",)),
            CaseResult("ex3", 1.0, "", ("extraction",)),
            CaseResult("r1", 0.30, "", ("refusal",)),
        ])
        assert run.mean_score >= 0.80
        gate = regression_gate(run, self.baseline(), tolerance=0.20)
        assert not gate
        assert any("refusal" in r for r in gate.reasons)

    def test_a_critical_case_regression_fails_at_zero_tolerance(self):
        # A safety refusal that worked yesterday and does not today is not a
        # rounding error and must not be averaged away.
        run = self.run_with([
            CaseResult("ex1", 1.0, "", ("extraction",)),
            CaseResult("safe1", 0.99, "", ("refusal",), critical=True),
        ])
        gate = regression_gate(run, self.baseline(), tolerance=0.5, per_tag_tolerance=0.5)
        assert not gate
        assert any("CRITICAL" in r for r in gate.reasons)

    def test_a_non_critical_case_may_wobble(self):
        run = self.run_with([CaseResult("ex1", 0.99, "", ("extraction",))])
        assert regression_gate(run, self.baseline(), tolerance=0.5, per_tag_tolerance=0.5)

    def test_a_tag_absent_from_the_run_is_not_treated_as_a_drop(self):
        # Removing cases is a change to the eval set, not a regression. If
        # this were scored as a drop to zero, every eval edit would fail CI.
        run = self.run_with([CaseResult("ex1", 1.0, "", ("extraction",))])
        gate = regression_gate(run, self.baseline(), tolerance=0.5)
        assert "refusal" not in " ".join(gate.reasons)

    def test_gate_result_is_truthy_and_falsy_as_expected(self):
        good = self.run_with([CaseResult("ex1", 1.0, "", ("extraction",))])
        assert bool(regression_gate(good, self.baseline(), tolerance=0.5, per_tag_tolerance=0.5))


class TestBaselineRoundTrip:
    def test_a_saved_baseline_reloads_and_gates_against_itself(self):
        run = EvalRun("v1", [
            CaseResult("a", 1.0, "", ("x",)),
            CaseResult("b", 0.5, "", ("y",), critical=True),
        ])
        b = run.to_baseline()
        # A run compared against itself must always pass, or the gate is
        # non-deterministic and no result it produces means anything.
        assert regression_gate(run, b)

    def test_baseline_survives_json(self, tmp_path):
        run = EvalRun("v1", [CaseResult("a", 1.0, "", ("x",))])
        path = tmp_path / "baseline.json"
        run.save_baseline(path)
        assert regression_gate(run, load_baseline(path))
        assert json.loads(path.read_text())["name"] == "v1"


class TestReport:
    def test_report_surfaces_critical_failures_prominently(self):
        run = EvalRun("v2", [CaseResult("safe1", 0.0, "", ("refusal",), critical=True)])
        text = format_report(run, regression_gate(run, {"mean_score": 1.0, "cases": {"safe1": 1.0}}))
        assert "CRITICAL FAILURES" in text
        assert "FAIL" in text
