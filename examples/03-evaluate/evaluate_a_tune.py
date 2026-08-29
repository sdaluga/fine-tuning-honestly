#!/usr/bin/env python3
"""
Example 03 — Evaluate a tune, and catch what it broke.
======================================================

    python examples/03-evaluate/evaluate_a_tune.py

No GPU. No API key. No network. Under a second.

WHAT THIS DEMONSTRATES
----------------------
A tune that improved exactly what it was asked to improve, and destroyed
something else on the way past. The headline number does not move at all:

    baseline    16/20 = 0.80
    after tune  16/20 = 0.80

Identical. A dashboard showing the aggregate would show a flat line, the tune
would ship, and the failure would surface weeks later in the cases nobody
thought to check.

The per-tag view is where it becomes visible, and the critical-case check is
what makes it non-negotiable. Both are the reason the gate has three checks
instead of one.

Run order matters here: this example comes BEFORE the tuning example on
purpose. Building the eval is not the step after training — it is the step
that makes training interpretable at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

from stub_models import BASE, GOOD_TUNE, NARROW_TUNE, make_model  # noqa: E402

from tuning_toolkit.evaluate import (  # noqa: E402
    Case,
    Scorer,
    contains,
    format_report,
    load_baseline,
    normalized_match,
    numeric_close,
    regression_gate,
    run_eval,
)

RULE = "=" * 74

SCORERS: dict[str, Scorer] = {
    # Numbers arrive as "4,182,000 MWh" and "11 hours". Exact string match
    # would fail every one of them and teach you nothing about correctness.
    "numeric": numeric_close(0.01),
    "normalized": normalized_match,
    # A refusal is judged on whether it refused, not on its wording.
    "contains": contains,
}


def load_cases() -> tuple[list[Case], dict[str, Scorer], dict[str, str]]:
    cases, per_case, id_by_prompt = [], {}, {}
    for line in (HERE / "cases.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cases.append(
            Case(
                id=r["id"],
                prompt=r["prompt"],
                expected=r["expected"],
                tags=tuple(r.get("tags", ())),
                critical=r.get("critical", False),
            )
        )
        per_case[r["id"]] = SCORERS[r["scorer"]]
        id_by_prompt[r["prompt"]] = r["id"]
    return cases, per_case, id_by_prompt


def evaluate(name: str, table: dict[str, str], cases, per_case, id_by_prompt):
    return run_eval(name, cases, make_model(table, id_by_prompt), scorers=per_case)


def main() -> int:
    cases, per_case, id_by_prompt = load_cases()

    print(RULE)
    print("EVALUATING A TUNE")
    print(RULE)
    print(f"  {len(cases)} cases · "
          f"{sum(1 for c in cases if c.critical)} marked critical · "
          f"tags: {', '.join(sorted({t for c in cases for t in c.tags}))}")

    # ---- 1. baseline ----------------------------------------------------
    base_run = evaluate("base-model", BASE, cases, per_case, id_by_prompt)
    baseline_path = HERE / "baselines" / "v1-base-model.json"
    baseline_path.parent.mkdir(exist_ok=True)
    base_run.save_baseline(baseline_path)

    print(f"\n{RULE}")
    print("1 · BASELINE — score this BEFORE you tune anything")
    print("-" * 74)
    print(format_report(base_run))
    print(f"\n  saved -> baselines/{baseline_path.name}")
    print("  Committed to the repository. A baseline that lives on somebody's")
    print("  laptop is not a baseline.")

    baseline = load_baseline(baseline_path)

    # ---- 2. the narrow tune ---------------------------------------------
    narrow = evaluate("narrow-tune", NARROW_TUNE, cases, per_case, id_by_prompt)
    gate = regression_gate(narrow, baseline)

    print(f"\n{RULE}")
    print("2 · AFTER TUNING FOR EXTRACTION")
    print("-" * 74)
    print(format_report(narrow, gate))

    print()
    print(f"  Aggregate: {baseline['mean_score']:.3f} -> {narrow.mean_score:.3f}")
    if abs(baseline["mean_score"] - narrow.mean_score) < 1e-9:
        print("  UNCHANGED. Every summary metric you would put on a dashboard")
        print("  says this tune did nothing. It did two things that cancelled.")

    print()
    print("  Per capability:")
    now = narrow.by_tag()
    for tag, was in sorted(baseline["by_tag"].items()):
        delta = now[tag] - was
        arrow = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else " ")
        print(f"    {tag:<16} {was:.3f} -> {now[tag]:.3f}   {arrow} {delta:+.3f}")

    print()
    print("  Extraction did exactly what was asked. Refusal collapsed, because")
    print("  the training set was all extraction and the model learned that its")
    print("  job is to answer the question in front of it — and a refusal is")
    print("  structurally the case where the right answer is not to.")

    if failed := narrow.failed_critical():
        print()
        print("  CRITICAL CASES THAT NOW FAIL:")
        for r in failed:
            case = next(c for c in cases if c.id == r.case_id)
            print(f"    [{r.case_id}] {case.prompt[:60]}")
            print(f"        model now says: {r.actual[:60]}")

    # ---- 3. the good tune ------------------------------------------------
    good = evaluate("good-tune", GOOD_TUNE, cases, per_case, id_by_prompt)
    good_gate = regression_gate(good, baseline)

    print(f"\n{RULE}")
    print("3 · THE SAME GAIN, WITHOUT THE DAMAGE")
    print("-" * 74)
    print(format_report(good, good_gate))
    print()
    print("  Identical extraction improvement, achieved with refusal examples")
    print("  kept in the training mix. Nothing regressed, so the gate passes.")
    print("  This is the version that ships.")

    # ---- verdict ---------------------------------------------------------
    print(f"\n{RULE}")
    print("VERDICT")
    print("-" * 74)
    print(f"  narrow tune : {'BLOCKED' if not gate else 'passed'}")
    print(f"  good tune   : {'BLOCKED' if not good_gate else 'passed'}")
    print()
    print("  Same aggregate score. Opposite decisions. That difference is the")
    print("  entire argument for building the eval before the tune rather than")
    print("  after it.")

    print(f"\n{RULE}")
    print("Read next: docs/02-evaluation-first.md")
    print(RULE)

    # Self-check. If the demonstration ever stops demonstrating — because a
    # scorer changed, or a case was edited — this example should fail loudly
    # rather than print a story that is no longer true.
    problems = []
    if abs(base_run.mean_score - narrow.mean_score) > 1e-9:
        problems.append("aggregate scores differ; the flat-average point no longer holds")
    if gate.passed:
        problems.append("the narrow tune was NOT blocked; the gate is not doing its job")
    if not good_gate.passed:
        problems.append("the good tune was blocked; the gate is too strict")
    if not narrow.failed_critical():
        problems.append("no critical case failed under the narrow tune")

    if problems:
        print("\nSELF-CHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
