"""
Evaluation — build this before you tune, or you are not tuning, you're hoping.
=============================================================================

The order that saves projects: EVAL FIRST, DATA SECOND, TUNING LAST.

Teams do it backwards. They tune, look at some outputs, decide it "seems
better", and ship. Then quality moves and nobody can say which change moved
it, because there was never a number.

An eval set does three jobs, and only the first is the obvious one:

  1. Tells you whether the tune helped.
  2. Tells you what it BROKE. This is the one people skip. Tuning for a
     narrow behaviour reliably degrades everything adjacent to it, and the
     degradation shows up in production, in the cases you didn't think to
     look at.
  3. Gives you the artifact a governance review actually asks for. "We
     evaluated it" is not reviewable. A committed eval set with a scored
     baseline is.

WHAT'S HERE
-----------
A small, dependency-free harness: cases, scorers, a run, and a REGRESSION
GATE that compares against a stored baseline and fails on a drop.

Nothing here calls a model. You supply the outputs — from a real inference
run, a fixture, or a stub. That keeps the harness itself testable, fast, and
free, and it means the scoring logic is covered by tests rather than being the
one uninspected thing standing between you and a shipping decision.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

#: A scorer maps (actual, expected) to [0.0, 1.0].
Scorer = Callable[[str, str], float]


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


def exact_match(actual: str, expected: str) -> float:
    return 1.0 if actual.strip() == expected.strip() else 0.0


def normalized_match(actual: str, expected: str) -> float:
    """Case- and whitespace-insensitive. The usual default for short answers."""
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()  # noqa: E731
    return 1.0 if norm(actual) == norm(expected) else 0.0


def contains(actual: str, expected: str) -> float:
    return 1.0 if expected.strip().lower() in actual.lower() else 0.0


def regex_match(pattern: str) -> Scorer:
    """Scorer factory. `expected` is ignored — the pattern is the spec."""
    compiled = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    return lambda actual, _expected: 1.0 if compiled.search(actual) else 0.0


def numeric_close(tolerance: float = 0.01) -> Scorer:
    """
    Compare the first number in each string within a relative tolerance.

    Exact string match on numbers is a classic false-negative factory: "1200",
    "1,200" and "1200.0" are the same answer and three different strings.
    """
    num = re.compile(r"-?\d[\d,]*\.?\d*")

    def score(actual: str, expected: str) -> float:
        a, e = num.search(actual), num.search(expected)
        if not a or not e:
            return 0.0
        try:
            av = float(a.group().replace(",", ""))
            ev = float(e.group().replace(",", ""))
        except ValueError:
            return 0.0
        if ev == 0:
            return 1.0 if abs(av) <= tolerance else 0.0
        return 1.0 if abs(av - ev) / abs(ev) <= tolerance else 0.0

    return score


def json_field(field_path: str) -> Scorer:
    """
    Compare one field of a JSON response, tolerating prose around the object.

    Models wrap JSON in fences and commentary. Scoring the whole string
    punishes the wrapper rather than the answer, which teaches you nothing
    about whether the model got it right.
    """
    def score(actual: str, expected: str) -> float:
        try:
            start, end = actual.index("{"), actual.rindex("}")
            data = json.loads(actual[start : end + 1])
        except (ValueError, json.JSONDecodeError):
            return 0.0
        cur = data
        for key in field_path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return 0.0
            cur = cur[key]
        return 1.0 if str(cur).strip().lower() == expected.strip().lower() else 0.0

    return score


# ---------------------------------------------------------------------------
# Cases and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """
    One evaluation case.

    `tags` is what makes job (2) above possible. Tag cases by capability and
    you can see that tuning lifted `extraction` by 0.2 while dropping
    `refusal` by 0.3 — a trade you may well accept, but only if you can see it.
    """

    id: str
    prompt: str
    expected: str
    tags: tuple[str, ...] = ()
    #: Cases that must never regress, whatever the aggregate does. Safety
    #: behaviours, refusals, regulatory language.
    critical: bool = False


@dataclass
class CaseResult:
    case_id: str
    score: float
    actual: str
    tags: tuple[str, ...] = ()
    critical: bool = False

    @property
    def passed(self) -> bool:
        return self.score >= 1.0


@dataclass
class EvalRun:
    name: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return statistics.mean(r.score for r in self.results) if self.results else 0.0

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def by_tag(self) -> dict[str, float]:
        """Mean score per tag. Where a regression actually becomes visible."""
        buckets: dict[str, list[float]] = {}
        for r in self.results:
            for tag in r.tags:
                buckets.setdefault(tag, []).append(r.score)
        return {tag: statistics.mean(v) for tag, v in sorted(buckets.items())}

    def failed_critical(self) -> list[CaseResult]:
        return [r for r in self.results if r.critical and not r.passed]

    def to_baseline(self) -> dict:
        """Serialise enough to compare a future run against this one."""
        return {
            "name": self.name,
            "mean_score": self.mean_score,
            "pass_rate": self.pass_rate,
            "by_tag": self.by_tag(),
            "cases": {r.case_id: r.score for r in self.results},
        }

    def save_baseline(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_baseline(), indent=2, sort_keys=True))


def run_eval(
    name: str,
    cases: Sequence[Case],
    generate: Callable[[str], str],
    scorer: Scorer = normalized_match,
    scorers: dict[str, Scorer] | None = None,
) -> EvalRun:
    """
    Score `generate` against `cases`.

    `scorers` overrides per case id, because one eval set legitimately mixes
    answer shapes — a number here, a JSON field there — and forcing them all
    through one scorer either loosens it into meaninglessness or fails cases
    that were right.
    """
    scorers = scorers or {}
    run = EvalRun(name=name)
    for case in cases:
        actual = generate(case.prompt)
        fn = scorers.get(case.id, scorer)
        run.results.append(
            CaseResult(
                case_id=case.id,
                score=fn(actual, case.expected),
                actual=actual,
                tags=case.tags,
                critical=case.critical,
            )
        )
    return run


# ---------------------------------------------------------------------------
# The regression gate
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


def regression_gate(
    run: EvalRun,
    baseline: dict,
    *,
    tolerance: float = 0.02,
    per_tag_tolerance: float = 0.05,
) -> GateResult:
    """
    Compare a run against a stored baseline and decide whether it ships.

    Three checks, in ascending order of how much they matter:

      1. Aggregate mean did not drop more than `tolerance`.
      2. NO TAG dropped more than `per_tag_tolerance`. This catches the
         characteristic tuning failure: overall flat, one capability quietly
         destroyed. The aggregate hides it by construction — that is what an
         average does.
      3. No critical case regressed AT ALL. Zero tolerance, deliberately.
         A safety refusal that worked yesterday and does not work today is
         not a rounding error, and it must not be averaged away.
    """
    reasons: list[str] = []

    drop = baseline.get("mean_score", 0.0) - run.mean_score
    if drop > tolerance:
        reasons.append(
            f"mean score dropped {drop:.3f} "
            f"({baseline.get('mean_score', 0.0):.3f} -> {run.mean_score:.3f}), "
            f"tolerance {tolerance}"
        )

    now = run.by_tag()
    for tag, was in baseline.get("by_tag", {}).items():
        if tag not in now:
            continue
        tag_drop = was - now[tag]
        if tag_drop > per_tag_tolerance:
            reasons.append(
                f"tag {tag!r} dropped {tag_drop:.3f} ({was:.3f} -> {now[tag]:.3f}), "
                f"tolerance {per_tag_tolerance} — the aggregate is hiding this"
            )

    base_cases = baseline.get("cases", {})
    for r in run.results:
        if r.critical and base_cases.get(r.case_id, 0.0) > r.score:
            reasons.append(
                f"CRITICAL case {r.case_id!r} regressed "
                f"({base_cases[r.case_id]:.2f} -> {r.score:.2f}) — zero tolerance"
            )

    return GateResult(passed=not reasons, reasons=reasons)


def load_baseline(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def format_report(run: EvalRun, gate: GateResult | None = None) -> str:
    """Human-readable summary for a terminal or a CI log."""
    lines = [
        f"eval: {run.name}",
        f"  cases      {len(run.results)}",
        f"  mean score {run.mean_score:.3f}",
        f"  pass rate  {run.pass_rate:.1%}",
    ]
    tags = run.by_tag()
    if tags:
        lines.append("  by tag:")
        lines += [f"    {t:<24} {s:.3f}" for t, s in tags.items()]
    if failed := run.failed_critical():
        lines.append(f"  CRITICAL FAILURES: {', '.join(r.case_id for r in failed)}")
    if gate is not None:
        lines.append(f"  gate: {'PASS' if gate.passed else 'FAIL'}")
        lines += [f"    - {r}" for r in gate.reasons]
    return "\n".join(lines)


def asdict_run(run: EvalRun) -> dict:
    return asdict(run)
