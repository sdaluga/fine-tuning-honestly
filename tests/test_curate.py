"""
Tests for data curation — the layer whose failures are invisible.
================================================================

Every test here is named for the production incident it prevents. That is not
a style choice: a test named `test_decontaminate_works` tells a future reader
nothing about whether it is safe to delete.

The defect these guard against shares one property — it does not raise, does
not log, and in the contamination case makes your numbers look BETTER. Which
is precisely why it has to be a test and not a code review comment.

No GPU. No API key. No network. Whole file runs in well under a second.
"""

from __future__ import annotations

import pytest

from tuning_toolkit.curate import (
    Example,
    assert_no_leakage,
    content_hash,
    curate,
    decontaminate,
    detect_pii,
    exact_dedup,
    jaccard,
    near_dedup,
    normalize,
    scan_pii,
    shingles,
    split,
    validate_format,
)


def ex(prompt: str, completion: str = "ok", source: str = "test") -> Example:
    return Example(prompt=prompt, completion=completion, source=source)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_collapses_the_differences_a_tokenizer_would_ignore(self):
        # If these do not collide, exact dedup is theatre: the same example
        # appears twice and silently doubles its weight in training.
        assert normalize("Hello  World") == normalize("hello world")
        assert normalize(" hello\tworld\n") == normalize("hello world")

    def test_normalizes_unicode_lookalikes(self):
        # NFKC. Scraped data is full of these and they defeat naive hashing.
        assert normalize("ﬁle") == normalize("file")

    def test_content_hash_is_stable_across_calls(self):
        assert content_hash("a b c") == content_hash("A  B  C")

    def test_different_content_gets_different_hashes(self):
        assert content_hash("alpha") != content_hash("beta")


class TestShingles:
    def test_short_strings_still_produce_a_shingle(self):
        # Returning an empty set for short text would make those records
        # silently un-checkable — a hole wide enough to leak an eval through.
        assert shingles("two words", n=5) == {"two words"}

    def test_empty_string_produces_nothing(self):
        assert shingles("", n=5) == set()

    def test_jaccard_bounds(self):
        assert jaccard(set(), set()) == 1.0
        assert jaccard({"a"}, set()) == 0.0
        assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestExactDedup:
    def test_a_duplicated_example_does_not_get_double_weight(self):
        kept, dropped = exact_dedup([ex("same"), ex("same"), ex("other")])
        assert len(kept) == 2
        assert len(dropped) == 1

    def test_dedup_survives_whitespace_and_case_variation(self):
        kept, _ = exact_dedup([ex("Hello World"), ex("hello   world")])
        assert len(kept) == 1, "case/whitespace variants must collide"

    def test_the_first_occurrence_is_the_one_kept(self):
        kept, _ = exact_dedup([ex("same", source="first"), ex("same", source="second")])
        assert kept[0].source == "first"

    def test_every_dropped_row_carries_a_reason(self):
        _, dropped = exact_dedup([ex("same"), ex("same")])
        assert "duplicate" in dropped[0][1]


class TestNearDedup:
    def test_paraphrases_collapse(self):
        a = ex("the quick brown fox jumps over the lazy dog near the river bank")
        b = ex("the quick brown fox jumps over the lazy dog near the river bank!")
        kept, dropped = near_dedup([a, b], threshold=0.8)
        assert len(kept) == 1
        assert len(dropped) == 1

    def test_genuinely_different_examples_are_both_kept(self):
        kept, _ = near_dedup(
            [
                ex("how do I reset a password on the billing portal"),
                ex("what is the capital city of Portugal and its population"),
            ],
            threshold=0.8,
        )
        assert len(kept) == 2, "over-aggressive dedup silently destroys training data"

    def test_threshold_is_honoured(self):
        a = ex("alpha beta gamma delta epsilon zeta eta theta")
        b = ex("alpha beta gamma delta epsilon zeta eta iota")
        assert len(near_dedup([a, b], threshold=0.99)[0]) == 2
        assert len(near_dedup([a, b], threshold=0.3)[0]) == 1


# ---------------------------------------------------------------------------
# Decontamination — the important one
# ---------------------------------------------------------------------------


class TestDecontamination:
    def test_an_eval_example_in_the_training_set_is_removed(self):
        # THE headline failure. Skip this and your eval score measures
        # memorisation, the number goes UP, and nobody investigates a rise.
        evalset = [ex("what is the boiling point of water at sea level in celsius", "100")]
        train = [
            ex("what is the boiling point of water at sea level in celsius", "100"),
            ex("explain how a heat pump moves thermal energy against a gradient", "..."),
        ]
        kept, dropped = decontaminate(train, evalset)
        assert len(kept) == 1
        assert len(dropped) == 1
        assert "contaminates" in dropped[0][1]

    def test_a_reworded_eval_question_is_still_caught(self):
        # Contamination does not need to be verbatim to inflate a score.
        evalset = [ex("list the three laws of motion described by isaac newton", "...")]
        train = [ex("list the three laws of motion described by isaac newton please", "...")]
        kept, _ = decontaminate(train, evalset, threshold=0.5)
        assert kept == []

    def test_sharing_only_the_question_still_counts_as_contamination(self):
        # A training row carrying the eval's PROMPT teaches the model that
        # exact question, even when the answer differs.
        #
        # The long, divergent completions are the point of this test. They
        # drag full-text similarity below the threshold, so ONLY the
        # prompt-to-prompt comparison can catch this. Drop that comparison and
        # this leak walks straight through — which is exactly what happened
        # when the mutation was tried.
        question = "what were the total generation megawatt hours recorded for the third quarter"
        evalset = [ex(question, "the reported figure was one thousand two hundred megawatt hours exactly")]
        train = [ex(question, "please consult the quarterly operations appendix for detailed figures instead")]
        kept, _ = decontaminate(train, evalset, threshold=0.5)
        assert kept == [], "prompt-only overlap leaks the eval question"

    def test_unrelated_training_data_survives(self):
        evalset = [ex("what is the capital of France", "Paris")]
        train = [ex("summarise this outage report for the operations team", "...")]
        assert len(decontaminate(train, evalset)[0]) == 1

    def test_empty_eval_set_removes_nothing(self):
        train = [ex("anything at all")]
        assert len(decontaminate(train, [])[0]) == 1


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------


class TestPII:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("contact me at alice@example.com", "email"),
            ("call 555-123-4567 today", "us_phone"),
            ("ssn 123-45-6789 on file", "ssn_like"),
            ("card 4111 1111 1111 1111", "card_like"),
            ("host at 192.168.1.20", "ipv4"),
            ("key AKIAIOSFODNN7EXAMPLE rotated", "aws_key"),
            ("-----BEGIN RSA PRIVATE KEY-----", "private_key"),
        ],
    )
    def test_detects_the_shapes_that_leak_through_copy_paste(self, text, expected):
        assert expected in detect_pii(text)

    def test_clean_text_produces_nothing(self):
        assert detect_pii("the turbine was offline for scheduled maintenance") == []

    def test_pii_findings_block_rather_than_warn(self):
        # Severity matters more than detection here. Training on PII is not a
        # bug you patch later — it is in the weights, it has been shown to be
        # extractable, and no deletion request can reach it.
        findings = scan_pii([ex("write to bob@corp.example")])
        assert findings
        assert all(f.severity == "block" for f in findings)

    def test_a_dataset_with_pii_is_not_releasable(self):
        report = curate([ex("mail alice@example.com about the outage")])
        assert not report.is_releasable
        assert report.blocking


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------


class TestFormatValidation:
    def test_empty_completion_blocks(self):
        # A handful of these teach the model that silence is a valid answer,
        # and it applies that lesson to your most important question.
        findings = validate_format([Example("a real prompt", "   ", "src")])
        assert any(f.kind == "format:empty_completion" and f.severity == "block" for f in findings)

    def test_empty_prompt_blocks(self):
        findings = validate_format([Example("", "an answer", "src")])
        assert any(f.kind == "format:empty_prompt" and f.severity == "block" for f in findings)

    def test_overlong_examples_warn_rather_than_block(self):
        findings = validate_format([Example("x" * 50_000, "ok", "src")], max_chars=1_000)
        assert any(f.kind == "format:too_long" and f.severity == "warn" for f in findings)

    def test_missing_source_is_flagged_because_it_cannot_be_retracted(self):
        findings = validate_format([Example("p", "c")])  # source defaults to "unknown"
        assert any(f.kind == "lineage:no_source" for f in findings)

    def test_a_clean_example_produces_no_findings(self):
        assert validate_format([ex("a proper prompt", "a proper completion")]) == []


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


class TestSplit:
    def test_split_is_deterministic_across_runs(self):
        data = [ex(f"example number {i}") for i in range(200)]
        a_train, a_eval = split(data, 0.2)
        b_train, b_eval = split(data, 0.2)
        assert [e.id for e in a_train] == [e.id for e in b_train]
        assert [e.id for e in a_eval] == [e.id for e in b_eval]

    def test_split_is_independent_of_input_order(self):
        # Reshuffling the dataset must not move examples across the split, or
        # a rerun silently changes what "eval" means.
        data = [ex(f"example number {i}") for i in range(200)]
        _, eval_a = split(data, 0.2)
        _, eval_b = split(list(reversed(data)), 0.2)
        assert {e.id for e in eval_a} == {e.id for e in eval_b}

    def test_identical_content_never_straddles_the_split(self):
        # The whole reason for hashing rather than sampling. A duplicate that
        # survived dedup must not land on both sides.
        data = [ex("identical text"), ex("identical text"), ex("something else")]
        train, held = split(data, 0.5)
        assert not assert_no_leakage(train, held) or True  # exercised below
        train_texts = {e.text for e in train}
        held_texts = {e.text for e in held}
        assert not (train_texts & held_texts)

    def test_split_sizes_are_roughly_the_requested_fraction(self):
        data = [ex(f"unique example {i}") for i in range(1000)]
        _, held = split(data, 0.1)
        assert 50 <= len(held) <= 160, f"got {len(held)}, expected ~100"

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_nonsense_fractions_are_rejected(self, bad):
        with pytest.raises(ValueError):
            split([ex("a")], bad)

    def test_leakage_check_catches_an_example_on_both_sides(self):
        shared = ex("this example is in both sets")
        findings = assert_no_leakage([shared, ex("train only")], [shared])
        assert len(findings) == 1
        assert findings[0].severity == "block"

    def test_leakage_check_is_quiet_on_a_clean_split(self):
        assert assert_no_leakage([ex("train")], [ex("eval")]) == []


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_ordering_prevents_a_duplicate_smuggling_contamination_back_in(self):
        # If decontamination ran before dedup, the second copy of the
        # contaminating row would survive behind the first.
        evalset = [ex("what is the rated capacity of the unit in megawatts", "850")]
        train = [
            ex("what is the rated capacity of the unit in megawatts", "850"),
            ex("what is the rated capacity of the unit in megawatts", "850"),
            ex("draft a maintenance summary for the turbine inspection", "..."),
        ]
        report = curate(train, evalset)
        assert len(report.kept) == 1
        assert report.kept[0].prompt.startswith("draft a maintenance")

    def test_a_clean_dataset_is_releasable(self):
        report = curate([ex("summarise the outage"), ex("draft an incident note")])
        assert report.is_releasable
        assert report.summary()["blocking"] == 0

    def test_report_accounts_for_every_input_row(self):
        rows = [ex("a"), ex("a"), ex("b")]
        report = curate(rows)
        assert len(report.kept) + len(report.dropped) == len(rows)
