"""
Tests for the governance artifacts.
===================================

The claim this file defends: a model card generated from the actual dataset,
at the moment of the actual run, is evidence — while one written afterwards is
a document. The difference is only real if the fingerprint is trustworthy, so
most of these tests are about the fingerprint.
"""

from __future__ import annotations

from datetime import datetime, timezone

from tuning_toolkit.curate import Example, curate
from tuning_toolkit.governance import (
    ModelCard,
    build_manifest,
    fingerprint_examples,
    reproducibility_check,
)


def ex(p: str, c: str = "ok", s: str = "corpus-a") -> Example:
    return Example(prompt=p, completion=c, source=s)


DATA = [ex("first prompt"), ex("second prompt"), ex("third prompt", s="corpus-b")]


def manifest_for(examples, checks=("dedup", "decontamination", "pii")):
    return build_manifest("test-set", "1.0", curate(examples), checks,
                          now=datetime(2026, 1, 1, tzinfo=timezone.utc))


class TestFingerprint:
    def test_the_same_dataset_fingerprints_identically(self):
        assert fingerprint_examples(DATA) == fingerprint_examples(list(DATA))

    def test_reordering_does_not_change_identity(self):
        # Sorted before hashing, deliberately. Without this every reshuffle
        # looks like a different dataset and the field becomes noise.
        assert fingerprint_examples(DATA) == fingerprint_examples(list(reversed(DATA)))

    def test_changing_one_example_changes_the_fingerprint(self):
        altered = DATA[:-1] + [ex("third prompt EDITED")]
        assert fingerprint_examples(altered) != fingerprint_examples(DATA)

    def test_removing_an_example_changes_the_fingerprint(self):
        assert fingerprint_examples(DATA[:-1]) != fingerprint_examples(DATA)

    def test_adding_an_example_changes_the_fingerprint(self):
        assert fingerprint_examples(DATA + [ex("new")]) != fingerprint_examples(DATA)

    def test_empty_dataset_has_a_stable_fingerprint(self):
        assert fingerprint_examples([]) == fingerprint_examples([])


class TestManifest:
    def test_records_what_was_checked_not_just_what_was_kept(self):
        # 10,000 clean rows means nothing alone. 10,000 rows that passed
        # decontamination and a PII scan is a claim someone can act on.
        m = manifest_for(DATA)
        assert m.checks_run == ["decontamination", "dedup", "pii"]

    def test_counts_examples_per_source_for_retraction(self):
        m = manifest_for(DATA)
        assert m.sources == {"corpus-a": 2, "corpus-b": 1}

    def test_summarises_findings_by_kind(self):
        m = manifest_for(DATA + [ex("mail alice@example.com")])
        assert any(k.startswith("pii:") for k in m.findings_summary)

    def test_serialises_to_json(self):
        assert '"fingerprint"' in manifest_for(DATA).to_json()


class TestModelCardGaps:
    def card(self, **kw) -> ModelCard:
        base = dict(
            model_name="triage-small",
            base_model="some-base-8b",
            version="1.0",
            owner="Platform AI",
            intended_use="Classify inbound operational messages by urgency.",
            method="lora",
            dataset=manifest_for(DATA),
        )
        base.update(kw)
        return ModelCard(**base)

    def test_a_bare_card_is_not_reviewable(self):
        card = self.card()
        assert not card.is_reviewable
        assert len(card.gaps()) >= 4

    def test_missing_evaluation_is_a_gap(self):
        assert any("eval" in g for g in self.card().gaps())

    def test_an_empty_limitations_list_is_a_gap(self):
        # Every model has limitations. An empty list tells a reviewer that
        # nobody looked, and they will be right.
        assert any("limitations" in g for g in self.card().gaps())

    def test_an_empty_out_of_scope_list_is_a_gap(self):
        # Without these the card licenses everything by omission.
        assert any("out-of-scope" in g for g in self.card().gaps())

    def test_missing_approval_is_a_gap(self):
        assert any("approval" in g for g in self.card().gaps())

    def test_a_complete_card_is_reviewable(self):
        card = self.card(
            eval_summary={"mean_score": 0.91, "refusal": 0.98},
            known_limitations=["Degrades on messages over 4k tokens."],
            out_of_scope_uses=["Any automated action without human review."],
            approvals=["Architecture review board, 2026-02-14"],
        )
        assert card.is_reviewable, card.gaps()

    def test_a_dataset_with_no_checks_is_a_gap(self):
        card = self.card(
            dataset=build_manifest("d", "1", curate(DATA), []),
            eval_summary={"mean_score": 0.9},
            known_limitations=["x"],
            out_of_scope_uses=["y"],
            approvals=["z"],
        )
        assert any("no checks" in g for g in card.gaps())


class TestModelCardMarkdown:
    def full_card(self) -> ModelCard:
        return ModelCard(
            model_name="triage-small",
            base_model="some-base-8b",
            version="1.0",
            owner="Platform AI",
            intended_use="Classify inbound operational messages by urgency.",
            method="lora",
            dataset=manifest_for(DATA),
            eval_summary={"mean_score": 0.91},
            known_limitations=["Degrades over 4k tokens."],
            out_of_scope_uses=["Automated action without review."],
            approvals=["ARB 2026-02-14"],
            hyperparameters={"rank": 16, "lr": 1e-4},
        )

    def test_renders_the_fields_a_reviewer_asks_for(self):
        md = self.full_card().to_markdown()
        for expected in ["Intended use", "Training data", "Fingerprint",
                         "Known limitations", "Out of scope", "Approvals"]:
            assert expected in md

    def test_an_unevaluated_model_is_marked_not_releasable_in_the_card_itself(self):
        card = self.full_card()
        card.eval_summary = {}
        assert "not releasable" in card.to_markdown()

    def test_gaps_are_printed_in_the_card_rather_than_hidden(self):
        card = self.full_card()
        card.approvals = []
        assert "Outstanding gaps" in card.to_markdown()

    def test_a_complete_card_prints_no_gap_section(self):
        assert "Outstanding gaps" not in self.full_card().to_markdown()


class TestReproducibility:
    def card_for(self, data) -> ModelCard:
        return ModelCard(
            model_name="m", base_model="b", version="1", owner="o",
            intended_use="u", method="lora", dataset=manifest_for(data),
        )

    def test_the_approved_dataset_verifies(self):
        assert reproducibility_check(self.card_for(DATA), DATA) == []

    def test_a_silently_edited_dataset_is_caught(self):
        # The question this answers — "is what I'm holding the thing that was
        # approved?" — gets asked six months later, during an incident, when
        # everyone's memory has become unreliable.
        card = self.card_for(DATA)
        tampered = DATA[:-1] + [ex("third prompt QUIETLY CHANGED")]
        problems = reproducibility_check(card, tampered)
        assert problems
        assert any("NOT the approved dataset" in p for p in problems)

    def test_a_shortened_dataset_reports_both_the_hash_and_the_count(self):
        problems = reproducibility_check(self.card_for(DATA), DATA[:-1])
        assert len(problems) == 2

    def test_reordering_the_dataset_still_verifies(self):
        # Order-independence has to hold end to end, or shuffling your
        # training data would read as tampering.
        assert reproducibility_check(self.card_for(DATA), list(reversed(DATA))) == []
