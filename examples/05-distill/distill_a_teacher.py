#!/usr/bin/env python3
"""
Example 05 — Distil a big model into a small one, without inheriting its bugs.
=============================================================================

    python examples/05-distill/distill_a_teacher.py

No GPU. No API key. No network. Under a second.

WHY THIS EXAMPLE EXISTS
-----------------------
Distillation is the most defensible reason to change weights — and the one
with a failure mode the others don't have, which is social as much as it is
technical:

    Teacher output gets treated as ground truth because it came from
    the expensive model.

It isn't. A teacher that is 80% accurate on your task hands you a label set
that is 80% accurate, and the student learns the other 20% as SYSTEMATIC
error — not as noise that partly washes out, because the teacher is wrong in
consistent ways.

This run shows two things, both with numbers:

  1. Sampling the teacher three times and keeping only what it agrees with
     itself on lifts label accuracy measurably, for one extra inference pass.
  2. Student-teacher AGREEMENT is the wrong headline metric. A student can
     agree with its teacher 100% of the time and be wrong on a fifth of the
     held-out set — and agreement is the number that ends up on the slide.

The teacher here is a fixture: twenty prompts, three sampled answers each,
with gold labels so the example can measure what filtering actually bought.
In a real run you would not have gold on the training set — that is the whole
reason you are distilling — which is exactly why the held-out set at the end
matters so much.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from tuning_toolkit.curate import curate  # noqa: E402
from tuning_toolkit.distill import (  # noqa: E402
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

RULE = "=" * 74


def load_candidates() -> list[Candidate]:
    rows = [
        json.loads(line)
        for line in (HERE / "teacher_samples.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return [
        Candidate(
            prompt=r["prompt"],
            samples=tuple(r["samples"]),
            gold=r.get("gold"),
            source="teacher:big-model-v2",
        )
        for r in rows
    ]


def main() -> int:
    cands = load_candidates()

    print(RULE)
    print("DISTILLING A TEACHER")
    print(RULE)
    print(f"  {len(cands)} prompts · 3 teacher samples each · gold labels held for scoring")

    # ---- 1. what the raw teacher output is actually worth ---------------
    raw_acc = label_accuracy(cands)
    print(f"\n{RULE}")
    print("1 · THE TEACHER IS NOT GROUND TRUTH")
    print("-" * 74)
    print(f"  raw label accuracy   {raw_acc:.3f}")
    print()
    print("  Train on all of this and the student learns the wrong ones too —")
    print("  systematically, because the teacher is wrong in consistent ways.")
    print("  Nothing about the file it arrived in says which is which.")

    # ---- 2. the cheapest quality lever there is -------------------------
    report = filter_by_consensus(cands)
    filtered_acc = label_accuracy(report.kept)

    print(f"\n{RULE}")
    print(f"2 · CONSENSUS FILTERING (keep only ≥ {DEFAULT_MIN_AGREEMENT:.2f} self-agreement)")
    print("-" * 74)
    for c, reason in report.dropped:
        _, ag = majority_vote(c.samples)
        mark = "wrong too" if c.gold and majority_vote(c.samples)[0] != c.gold else "was correct"
        print(f"  dropped [{ag:.2f}]  {c.prompt[:44]:<44} ({mark})")

    print()
    print(f"  kept {len(report.kept)} of {len(cands)}")
    print(f"  label accuracy   {raw_acc:.3f}  ->  {filtered_acc:.3f}"
          f"   ({filtered_acc - raw_acc:+.3f})")
    print()
    print("  One extra inference pass over the training set. That is the whole")
    print("  cost, and it is far cheaper than the tuning run it protects.")
    print("  Note the false positive above: filtering also discards some")
    print("  correct labels. Smaller and cleaner beats larger and noisier.")

    # ---- 3. machine-generated still goes through curation ---------------
    examples = to_examples(report.kept)
    cur = curate(examples)
    print(f"\n{RULE}")
    print("3 · MACHINE-GENERATED DOES NOT MEAN CLEAN")
    print("-" * 74)
    print(f"  {len(examples)} teacher-labelled examples -> curate()")
    print(f"  kept {len(cur.kept)} · dropped {len(cur.dropped)} · blocking {len(cur.blocking)}")
    for ex, reason in cur.dropped:
        print(f"    dropped: {reason}")
    for f in cur.blocking:
        print(f"    BLOCK [{f.kind}] {f.detail}")
    print()
    print("  Teacher output duplicates, contaminates and leaks PII exactly like")
    print("  human-written data — more so, because it was produced in bulk.")
    print(f"  Every surviving row is tagged source={examples[0].source!r}, so")
    print("  'which rows were machine-labelled' is a filter, not an investigation.")

    # ---- 4. the agreement trap ------------------------------------------
    # Held-out set the teacher never labelled. This is the only honest score.
    held = json.loads((HERE / "heldout.json").read_text())
    gold = held["gold"]
    teacher_answers = held["teacher"]
    student_answers = held["student"]

    fidelity = agreement_rate(student_answers, teacher_answers)
    student_acc = accuracy_against_gold(student_answers, gold)
    teacher_acc = accuracy_against_gold(teacher_answers, gold)

    print(f"\n{RULE}")
    print("4 · THE AGREEMENT TRAP")
    print("-" * 74)
    print(f"  student agrees with teacher   {fidelity:.3f}   <- the flattering number")
    print(f"  teacher accuracy vs gold      {teacher_acc:.3f}")
    print(f"  STUDENT ACCURACY vs gold      {student_acc:.3f}   <- the real one")
    print()
    print("  The student reproduces its teacher almost perfectly. That is what")
    print("  distillation is for, and it is not evidence the model is good —")
    print("  it is evidence the copy worked. Agreement measures FIDELITY.")
    print("  Only gold measures quality, and only on data the teacher never saw.")

    # ---- 5. the ceiling ---------------------------------------------------
    floor_raw = expected_accuracy(raw_acc, fidelity)
    floor_filtered = expected_accuracy(filtered_acc, fidelity)

    print(f"\n{RULE}")
    print("5 · THE CEILING")
    print("-" * 74)
    print(f"  with raw labels        floor ≈ {raw_acc:.3f} × {fidelity:.3f} = {floor_raw:.3f}")
    print(f"  with filtered labels   floor ≈ {filtered_acc:.3f} × {fidelity:.3f} = {floor_filtered:.3f}")
    print()
    print("  A student that perfectly reproduces its labels is exactly as")
    print("  accurate as they are. Distillation cannot exceed its teacher on")
    print("  the labels it was given — so if you need better than the teacher,")
    print("  the lever is better labels, not more epochs.")
    print()
    print("  (That product is a LOWER bound: it ignores the student getting")
    print("   lucky by disagreeing with a wrong label. It is also symmetric in")
    print("   its two terms, so it cannot tell you which lever to pull first.)")

    print(f"\n{RULE}")
    print("Read next: docs/04-tuning-methods.md")
    print(RULE)

    # Self-check, same discipline as example 03: if the demonstration stops
    # demonstrating, fail rather than print a story that is no longer true.
    problems = []
    if filtered_acc <= raw_acc:
        problems.append("consensus filtering did not improve label accuracy")
    if fidelity <= student_acc:
        problems.append("agreement is not higher than accuracy; the trap is not shown")
    if not cur.dropped and not cur.blocking:
        problems.append("curation found nothing in the teacher output")

    if problems:
        print("\nSELF-CHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
