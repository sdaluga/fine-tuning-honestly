"""
Tests for the tune-or-don't decision model.
===========================================

The recommendation logic is checked cheapest-rung-first, and these tests pin
that ordering down. The reason it matters: a scenario that is really a
retrieval problem but also mentions cost must not fall through into a tuning
recommendation. Reorder the checks and it will — silently, and in the
direction of the more expensive answer, which is the direction people are
already biased toward.
"""

from __future__ import annotations

import pytest

from tuning_toolkit.decision import (
    MIN_PROMPT_ITERATIONS,
    MIN_SFT_EXAMPLES,
    CostInputs,
    Rung,
    Scenario,
    breakeven_months,
    cost_summary,
    monthly_cost_prompted,
    monthly_cost_tuned,
    recommend,
)


def ready(**kw) -> Scenario:
    """A scenario with every precondition for tuning already satisfied."""
    base = dict(has_eval_set=True, prompt_iterations=5, labelled_examples=5_000)
    base.update(kw)
    return Scenario(**base)


class TestTheLadderIsOrdered:
    def test_rungs_ascend_in_cost(self):
        assert Rung.PROMPT < Rung.RETRIEVAL < Rung.DISTILL < Rung.SFT < Rung.CONTINUED_PT

    def test_weight_changing_starts_at_distillation(self):
        assert not recommend(Scenario(failure_is_missing_knowledge=True)).changes_weights
        assert recommend(ready(quality_is_fine_cost_is_not=True)).changes_weights


class TestMisdiagnosisIsCaughtFirst:
    def test_missing_knowledge_routes_to_retrieval_not_tuning(self):
        # The single most common misdiagnosis. Weights are a bad place for
        # facts: they go stale, cannot be cited, and correcting one means a
        # retrain.
        r = recommend(ready(failure_is_missing_knowledge=True))
        assert r.rung == Rung.RETRIEVAL
        assert not r.changes_weights

    def test_missing_knowledge_wins_even_when_cost_is_also_a_problem(self):
        # THE ordering test. Both flags set; the cheaper diagnosis must win.
        r = recommend(ready(failure_is_missing_knowledge=True, quality_is_fine_cost_is_not=True))
        assert r.rung == Rung.RETRIEVAL, "a retrieval problem must not fall through to tuning"

    def test_missing_knowledge_wins_over_a_behaviour_claim(self):
        r = recommend(ready(failure_is_missing_knowledge=True, failure_is_behaviour=True))
        assert r.rung == Rung.RETRIEVAL

    def test_missing_access_routes_to_tools(self):
        r = recommend(ready(failure_is_missing_access=True))
        assert r.rung == Rung.TOOLS
        assert not r.changes_weights


class TestLegitimateReasonsToChangeWeights:
    def test_cost_pressure_routes_to_distillation(self):
        r = recommend(ready(quality_is_fine_cost_is_not=True))
        assert r.rung == Rung.DISTILL
        assert r.is_actionable

    def test_behaviour_that_survives_prompting_routes_to_sft(self):
        r = recommend(ready(failure_is_behaviour=True))
        assert r.rung == Rung.SFT

    def test_unstateable_taste_routes_to_preference_tuning(self):
        r = recommend(ready(failure_is_unstateable_taste=True, labelled_examples=5_000))
        assert r.rung == Rung.PREFERENCE

    def test_preference_tuning_is_blocked_without_enough_pairs(self):
        r = recommend(ready(failure_is_unstateable_taste=True, labelled_examples=MIN_SFT_EXAMPLES))
        assert r.rung == Rung.PREFERENCE
        assert not r.is_actionable
        assert any("ranked pairs" in b for b in r.blockers)

    def test_novel_domain_routes_to_continued_pretraining(self):
        r = recommend(ready(domain_is_novel=True))
        assert r.rung == Rung.CONTINUED_PT


class TestTheDefaultIsNotToTune:
    def test_an_empty_scenario_recommends_prompting(self):
        r = recommend(Scenario())
        assert r.rung == Rung.PROMPT
        assert not r.changes_weights

    def test_iterated_prompting_graduates_to_curated_few_shot(self):
        r = recommend(Scenario(prompt_iterations=MIN_PROMPT_ITERATIONS))
        assert r.rung == Rung.FEW_SHOT

    def test_no_stated_failure_never_recommends_weights(self):
        # Having data and an eval set is not, by itself, a reason to tune.
        r = recommend(ready())
        assert not r.changes_weights


class TestBlockersAreNotAdvisory:
    def test_no_eval_set_blocks_every_weight_changing_rung(self):
        r = recommend(ready(failure_is_behaviour=True, has_eval_set=False))
        assert not r.is_actionable
        assert any("eval set" in b for b in r.blockers)

    def test_too_few_prompt_iterations_blocks(self):
        r = recommend(ready(failure_is_behaviour=True, prompt_iterations=1))
        assert any("prompt iterations" in b for b in r.blockers)

    def test_too_little_data_blocks(self):
        r = recommend(ready(failure_is_behaviour=True, labelled_examples=10))
        assert any("labelled examples" in b for b in r.blockers)

    def test_a_blocked_recommendation_still_names_the_right_rung(self):
        # "Not yet" is different from "not this". The rung is still the
        # correct diagnosis; the blockers are what stands in the way.
        r = recommend(Scenario(failure_is_behaviour=True))
        assert r.rung == Rung.SFT
        assert not r.is_actionable

    def test_cheap_rungs_are_never_blocked(self):
        # Prompting has no preconditions. Gating it would be absurd.
        assert recommend(Scenario()).is_actionable

    def test_every_recommendation_carries_a_reason(self):
        for s in [Scenario(), ready(failure_is_behaviour=True), ready(failure_is_missing_access=True)]:
            assert len(recommend(s).reason) > 40


class TestCostModel:
    def base(self, **kw) -> CostInputs:
        d = dict(
            monthly_requests=1_000_000,
            input_tokens=800,
            output_tokens=200,
            large_in_per_m=3.0,
            large_out_per_m=15.0,
            small_in_per_m=0.25,
            small_out_per_m=1.25,
            tuning_setup_cost=40_000.0,
            tuning_monthly_overhead=2_000.0,
        )
        d.update(kw)
        return CostInputs(**d)

    def test_prompted_cost_is_arithmetic_we_can_check_by_hand(self):
        c = self.base()
        # 1M requests * (800 * 3 + 200 * 15) / 1M = 800*3 + 200*15 = 5400
        assert monthly_cost_prompted(c) == pytest.approx(5_400.0)

    def test_tuned_cost_includes_the_overhead_people_forget(self):
        c = self.base()
        # 800*0.25 + 200*1.25 = 450, plus 2000 overhead
        assert monthly_cost_tuned(c) == pytest.approx(2_450.0)

    def test_breakeven_is_setup_divided_by_saving(self):
        c = self.base()
        assert breakeven_months(c) == pytest.approx(40_000 / 2_950, rel=1e-6)

    def test_low_volume_never_breaks_even(self):
        # The most useful output in the module. At 10k requests a month the
        # ownership overhead alone exceeds the entire per-token saving, so
        # tuning never pays back — no matter how patient you are.
        c = self.base(monthly_requests=10_000)
        assert breakeven_months(c) is None

    def test_overhead_alone_can_destroy_the_case(self):
        c = self.base(tuning_monthly_overhead=10_000.0)
        assert breakeven_months(c) is None, "overhead exceeding the saving must return None"

    def test_summary_exposes_the_none(self):
        s = cost_summary(self.base(monthly_requests=10_000))
        assert s["breakeven_months"] is None
        assert s["monthly_saving"] < 0

    def test_higher_volume_shortens_payback(self):
        assert breakeven_months(self.base(monthly_requests=5_000_000)) < breakeven_months(self.base())
