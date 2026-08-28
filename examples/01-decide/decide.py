#!/usr/bin/env python3
"""
Example 01 — Should we tune? Four real scenarios, four different answers.
=========================================================================

    python examples/01-decide/decide.py

No GPU. No API key. No network. Runs in about a second.

The point of this example is that three of these four scenarios sound, in a
planning meeting, like fine-tuning problems. Only one of them is. The other
three are a retrieval problem, an access problem, and an economics problem
wearing a quality problem's clothes.

Every scenario below is a shape I have watched teams argue about. The numbers
are illustrative; the failure modes are not.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tuning_toolkit.decision import (  # noqa: E402
    CostInputs,
    Scenario,
    cost_summary,
    recommend,
)

SCENARIOS: list[tuple[str, str, Scenario]] = [
    (
        "Outage assistant gets the facts wrong",
        "Engineers ask it about equipment it describes confidently and incorrectly. "
        "The right answers live in a maintenance system that changes daily.",
        Scenario(
            failure_is_missing_knowledge=True,
            prompt_iterations=6,
            labelled_examples=3_000,
            has_eval_set=True,
            monthly_requests=120_000,
        ),
    ),
    (
        "Assistant can't answer 'what is the status right now'",
        "It reasons well but has no path to the system of record, so it guesses "
        "or refuses.",
        Scenario(
            failure_is_missing_access=True,
            prompt_iterations=8,
            labelled_examples=1_200,
            has_eval_set=True,
            monthly_requests=90_000,
        ),
    ),
    (
        "Classification is good; the bill is not",
        "A frontier model classifies inbound messages at the accuracy the "
        "business wants. At four million messages a month, the invoice is the "
        "entire objection.",
        Scenario(
            quality_is_fine_cost_is_not=True,
            prompt_iterations=10,
            labelled_examples=40_000,
            has_eval_set=True,
            monthly_requests=4_000_000,
        ),
    ),
    (
        "Reports never match house style, and prompting hasn't fixed it",
        "Twelve prompt revisions. The structure and register are still wrong in "
        "a way reviewers can point at but nobody can write down as a rule.",
        Scenario(
            failure_is_behaviour=True,
            prompt_iterations=12,
            labelled_examples=2_400,
            has_eval_set=True,
            monthly_requests=8_000,
        ),
    ),
]


def show(title: str, blurb: str, scenario: Scenario) -> None:
    rec = recommend(scenario)

    print(f"\n{'=' * 74}")
    print(title)
    print("-" * 74)
    print(f"  {blurb}")
    print()
    print(f"  RECOMMENDATION   rung {int(rec.rung)} — {rec.rung.name}")
    print(f"  Changes weights  {'yes' if rec.changes_weights else 'no'}")
    print(f"  Actionable now   {'yes' if rec.is_actionable else 'NO — see blockers'}")
    print()
    print("  Why:")
    for line in _wrap(rec.reason, 68):
        print(f"    {line}")

    if rec.try_first:
        print()
        print(f"  Exhaust first:   {', '.join(r.name for r in rec.try_first)}")

    if rec.blockers:
        print()
        print("  BLOCKERS — tuning is not yet a coherent option:")
        for b in rec.blockers:
            for i, line in enumerate(_wrap(b, 64)):
                print(f"    {'- ' if i == 0 else '  '}{line}")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def show_cost() -> None:
    """
    The economics behind scenario 3, and the same model at low volume.

    The second case is the one worth staring at. Identical per-token
    advantage, identical setup cost — and it never pays back, because the
    monthly cost of OWNING a tuned model eats the whole saving. That line is
    the one missing from most business cases.
    """
    print(f"\n{'=' * 74}")
    print("The economics — same tuning, two volumes")
    print("-" * 74)

    def row(label: str, reqs: int) -> None:
        c = CostInputs(
            monthly_requests=reqs,
            input_tokens=900,
            output_tokens=150,
            large_in_per_m=3.00,
            large_out_per_m=15.00,
            small_in_per_m=0.20,
            small_out_per_m=0.80,
            tuning_setup_cost=45_000.0,      # overwhelmingly human time
            tuning_monthly_overhead=3_500.0,  # evals, drift, re-tunes
        )
        s = cost_summary(c)
        be = s["breakeven_months"]
        be_txt = f"{be:.1f} months" if be is not None else "NEVER"
        print(f"\n  {label}  ({reqs:,} requests/month)")
        print(f"    prompted, large model   ${s['monthly_prompted']:>12,.0f} / month")
        print(f"    tuned, small model      ${s['monthly_tuned']:>12,.0f} / month")
        print(f"    monthly saving          ${s['monthly_saving']:>12,.0f}")
        print(f"    one-time setup          ${s['setup_cost']:>12,.0f}")
        print(f"    breakeven               {be_txt:>13}")

    row("High volume", 4_000_000)
    row("Low volume ", 25_000)

    print()
    print("  The low-volume case is the finding. Same per-token advantage,")
    print("  same setup cost — and it never pays back, because $3,500/month of")
    print("  ownership overhead exceeds the entire saving. Tuning did not fail")
    print("  here; it was never going to work, and thirty seconds of arithmetic")
    print("  says so before anyone spends a quarter finding out.")


def main() -> int:
    print("=" * 74)
    print("SHOULD WE FINE-TUNE?")
    print("=" * 74)
    print("Four scenarios. Three sound like tuning problems in a planning")
    print("meeting. One is.")

    for title, blurb, scenario in SCENARIOS:
        show(title, blurb, scenario)

    show_cost()

    print(f"\n{'=' * 74}")
    print("Read next: docs/01-when-to-tune.md")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
