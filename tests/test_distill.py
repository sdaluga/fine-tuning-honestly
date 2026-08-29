"""
Tests for distillation — the labels came from a model and nobody checked them.
=============================================================================

The failure this file guards against is social before it is technical:
teacher output gets treated as ground truth because it came from the
expensive model. It isn't. A 90%-accurate teacher produces a 90%-accurate
label set, and the student learns the other 10% as systematic error.
"""

from __future__ import annotations

import pytest

from tuning_toolkit.curate import curate
from tuning_toolkit.distill import (
    DEFAULT_MIN_AGREEMENT,
    Candidate,
    accuracy_against_gold,
    agreement_rate,
    expected_accuracy,
    filter_by_consensus,
    label_accuracy,
    majority_vote,
    to_examples,
)


def cand(prompt: str, *samples: str, gold: str | None = None) -> Candidate:
    return Candidate(prompt=prompt, samples=tuple(samples), gold=gold)


class TestMajorityVote:
    def test_unanimous_teacher_scores_one(self):
        assert majority_vote(["now", "now", "now"]) == ("now", 1.0)

    def test_two_of_three_scores_two_thirds(self):
        label, agreement = majority_vote(["now", "now", "today"])
        assert label == "now"
        assert agreement == pytest.approx(2 / 3)

    def test_three_way_disagreement_scores_one_third(self):
        _, agreement = majority_vote(["now", "today", "this_week"])
        assert agreement == pytest.approx(1 / 3)

    def test_votes_are_counted_on_normalised_text(self):
        # "Now", "now " and "NOW" are one answer given three times, not three
        # answers. Counting them separately would make a confident teacher
        # look uncertain and throw away its best labels.
        label, agreement = majority_vote(["Now", "now ", "NOW"])
        assert agreement == 1.0
        assert label == "Now", "the original spelling is what gets trained on"

    def test_returns_an_original_sample_not_a_normalised_one(self):
        label, _ = majority_vote(["  Urgency: NOW  ", "urgency: now", "today"])
        assert label in ("  Urgency: NOW  ", "urgency: now")

    def test_ties_are_deterministic(self):
        first = majority_vote(["a", "b"])
        for _ in range(20):
            assert majority_vote(["a", "b"]) == first

    def test_empty_samples_are_rejected_at_construction(self):
        with pytest.raises(ValueError):
            Candidate(prompt="p", samples=())


class TestConsensusFiltering:
    def test_an_inconsistent_teacher_answer_is_dropped(self):
        report = filter_by_consensus([cand("p1", "now", "today", "this_week")])
        assert report.kept == []
        assert "self-agreement" in report.dropped[0][1]

    def test_a_consistent_answer_survives(self):
        report = filter_by_consensus([cand("p1", "now", "now", "now")])
        assert len(report.kept) == 1

    def test_two_of_three_passes_the_default_threshold(self):
        # This is the test that caught the original 0.67 default: 2/3 is
        # 0.6666…, so a hard `<` against 0.67 rejected exactly the case the
        # default was named for.
        report = filter_by_consensus([cand("p1", "now", "now", "today")])
        assert len(report.kept) == 1

    def test_the_default_is_exactly_two_of_three(self):
        assert DEFAULT_MIN_AGREEMENT == pytest.approx(2 / 3)

    def test_raising_the_threshold_trades_size_for_quality(self):
        cands = [cand("p1", "now", "now", "today"), cand("p2", "now", "now", "now")]
        assert len(filter_by_consensus(cands).kept) == 2
        assert len(filter_by_consensus(cands, 1.0).kept) == 1

    def test_filtering_actually_improves_label_accuracy(self):
        # THE claim this module makes. Not "filtering feels prudent" —
        # filtering measurably raises the accuracy of the labels you train on.
        cands = [
            cand("p1", "now", "now", "now", gold="now"),            # confident, right
            cand("p2", "today", "today", "today", gold="today"),    # confident, right
            cand("p3", "now", "now", "today", gold="now"),          # mostly right
            cand("p4", "now", "today", "this_week", gold="today"),  # unsure AND wrong
            cand("p5", "today", "now", "no_action", gold="now"),    # unsure AND wrong
        ]
        before = label_accuracy(cands)
        after = label_accuracy(filter_by_consensus(cands).kept)

        assert before == pytest.approx(0.6)
        assert after == pytest.approx(1.0)
        assert after > before, "consensus filtering must improve label quality"

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_nonsense_thresholds_are_rejected(self, bad):
        with pytest.raises(ValueError):
            filter_by_consensus([cand("p", "a")], bad)

    def test_every_candidate_is_accounted_for(self):
        cands = [cand("p1", "a", "a"), cand("p2", "a", "b"), cand("p3", "c", "c")]
        r = filter_by_consensus(cands)
        assert len(r.kept) + len(r.dropped) == len(cands)


class TestLabelAccuracy:
    def test_returns_none_without_gold_labels(self):
        # The honest answer. In the normal case you have no gold — if you did,
        # you would not be distilling — and a comforting number computed from
        # nothing is worse than an absent one.
        assert label_accuracy([cand("p1", "now", "now")]) is None

    def test_scores_only_the_candidates_that_have_gold(self):
        cands = [cand("p1", "now", "now", gold="now"), cand("p2", "today", "today")]
        assert label_accuracy(cands) == 1.0


class TestTheAgreementTrap:
    def test_high_agreement_can_coexist_with_poor_accuracy(self):
        # The trap, in one test. The student reproduces the teacher perfectly.
        # The teacher is wrong on two of five. Agreement says 1.00 and the
        # student is 0.60 accurate — and 1.00 is the number that ends up on
        # the slide.
        teacher = {"p1": "now", "p2": "today", "p3": "now", "p4": "now", "p5": "today"}
        student = dict(teacher)
        gold = {"p1": "now", "p2": "today", "p3": "now", "p4": "today", "p5": "now"}

        assert agreement_rate(student, teacher) == 1.0
        assert accuracy_against_gold(student, gold) == pytest.approx(0.6)

    def test_a_student_that_diverges_can_be_better_than_its_teacher(self):
        # Rarer, and worth knowing: lower agreement is not automatically bad.
        # It is only bad if accuracy fell with it.
        teacher = {"p1": "now", "p2": "wrong"}
        student = {"p1": "now", "p2": "right"}
        gold = {"p1": "now", "p2": "right"}

        assert agreement_rate(student, teacher) == pytest.approx(0.5)
        assert accuracy_against_gold(student, gold) == 1.0

    def test_no_overlap_scores_zero_rather_than_dividing_by_zero(self):
        assert agreement_rate({"a": "1"}, {"b": "1"}) == 0.0
        assert accuracy_against_gold({"a": "1"}, {"b": "1"}) == 0.0


class TestTheCeiling:
    def test_perfect_fidelity_means_the_student_inherits_label_accuracy(self):
        # The consequence people miss: distillation cannot exceed its teacher
        # on the labels it was given.
        assert expected_accuracy(0.85, 1.0) == pytest.approx(0.85)

    def test_imperfect_fidelity_lowers_the_floor_further(self):
        assert expected_accuracy(0.90, 0.95) == pytest.approx(0.855)

    def test_the_bound_is_symmetric_and_therefore_cannot_rank_the_two_levers(self):
        # Kept as documentation of a limitation rather than deleted quietly.
        # An early draft claimed five points of label accuracy beat five points
        # of fidelity. It does not follow from this model: `p * f` is
        # symmetric, so the two are identical here. Ranking the levers needs
        # the real error distribution.
        assert expected_accuracy(0.95, 0.90) == pytest.approx(expected_accuracy(0.90, 0.95))

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_out_of_range_inputs_are_rejected(self, bad):
        with pytest.raises(ValueError):
            expected_accuracy(bad, 0.9)
        with pytest.raises(ValueError):
            expected_accuracy(0.9, bad)


class TestHandoffToCuration:
    def test_teacher_labels_still_go_through_the_normal_pipeline(self):
        # Machine-generated does not mean clean. Teacher output duplicates,
        # contaminates and leaks PII exactly like human-written data — and
        # more so, because it was produced in bulk.
        cands = [
            cand("Summarise the outage report", "The unit tripped on low lube-oil pressure."),
            cand("Summarise the outage report", "The unit tripped on low lube-oil pressure."),
            cand("Who is the site contact", "Dana Whitfield, dana.whitfield@example.com"),
        ]
        report = curate(to_examples(cands))

        assert len(report.kept) == 2, "duplicate teacher output must still be deduped"
        assert not report.is_releasable, "PII in teacher output must still block"

    def test_source_records_that_a_model_wrote_the_label(self):
        examples = to_examples([Candidate("p", ("a", "a"), source="teacher:big-model-v2")])
        assert examples[0].source == "teacher:big-model-v2"
