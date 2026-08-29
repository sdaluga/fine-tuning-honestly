"""
Distillation — where the labels come from a model, and nobody checks them.
=========================================================================

Distillation is the most defensible reason to change weights: quality is
already fine, cost is not, and you teach a small model to imitate a large one
on ONE narrow task. That is a far easier target than general capability, and
it is the most common genuine win in enterprise settings.

It also has a failure mode the other methods don't, and it is a social one as
much as a technical one:

    Teacher output gets treated as ground truth because it came from
    the expensive model.

It is not ground truth. It is one model's answer, and a model that is 90%
accurate on your task produces a label set that is 90% accurate. Train on it
uncritically and the student learns the 10% too — not as random noise, which
would partly wash out, but as SYSTEMATIC error, because the teacher is wrong
in consistent ways. You have distilled the mistakes with the same fidelity as
everything else.

TWO THINGS THIS MODULE EXISTS TO STOP
-------------------------------------
1. THE AGREEMENT TRAP. The natural metric for distillation is student-teacher
   agreement, and it is the wrong one. Agreement with a wrong teacher is a
   wrong student that scores 0.98. Measure the student against ground truth,
   on a held-out set the teacher never labelled. If you have no such set, you
   do not know whether distillation worked — you only know it copied.

2. UNFILTERED TEACHER OUTPUT. Sampling the teacher more than once and keeping
   only the answers it agrees with itself on is the cheapest quality lever in
   the whole pipeline. It costs k× inference on the training set, once, and
   it is far cheaper than the tuning run it protects.

Everything here is pure and runs with no GPU, no API key and no network. You
supply the teacher's samples; how you obtained them is your business.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .curate import Example, normalize


@dataclass(frozen=True)
class Candidate:
    """
    One prompt, with k independent samples of the teacher's answer.

    `gold` is the true answer, and it is `None` in the normal case — if you
    had gold labels for everything you would not be distilling. It exists so
    that a held-out slice can be scored honestly, and so that this module's
    own tests can measure what filtering actually bought.
    """

    prompt: str
    samples: tuple[str, ...]
    gold: str | None = None
    source: str = "teacher"

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError(f"candidate for {self.prompt[:40]!r} has no teacher samples")


@dataclass
class DistillReport:
    kept: list[Candidate] = field(default_factory=list)
    dropped: list[tuple[Candidate, str]] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {"kept": len(self.kept), "dropped": len(self.dropped)}


# ---------------------------------------------------------------------------
# Self-consistency
# ---------------------------------------------------------------------------


def majority_vote(samples: Sequence[str]) -> tuple[str, float]:
    """
    The modal answer and the fraction of samples that agree with it.

    Comparison is on normalised text, but the RETURNED label is the original
    spelling of a winning sample — you want to train on what the teacher
    actually wrote, not on a lowercased, whitespace-collapsed version of it.

    Ties break toward the first sample seen, which keeps the function
    deterministic. A tie is a 0.5-or-lower agreement score anyway, so any
    sane threshold has already rejected it.
    """
    if not samples:
        raise ValueError("no samples")

    counts = Counter(normalize(s) for s in samples)
    winner_norm, n = counts.most_common(1)[0]
    original = next(s for s in samples if normalize(s) == winner_norm)
    return original, n / len(samples)


#: "At least 2 of 3". Written as a fraction rather than 0.67 because 2/3 is
#: 0.6666… and a hard `<` against 0.67 rejects exactly the case the default is
#: named for. A test caught that; the comparison below carries an epsilon for
#: the same reason.
DEFAULT_MIN_AGREEMENT = 2 / 3

_EPS = 1e-9


def filter_by_consensus(
    candidates: Sequence[Candidate], min_agreement: float = DEFAULT_MIN_AGREEMENT
) -> DistillReport:
    """
    Keep only the candidates the teacher agrees with itself on.

    THE CHEAPEST QUALITY LEVER IN THE PIPELINE.

    The premise is simple and empirically reliable: a model that samples the
    same answer three times out of three is more often right than one that
    gives three different answers. Self-consistency is not a correctness
    oracle — the teacher can be confidently and consistently wrong — but it
    is strongly correlated with correctness, and it costs k× inference on the
    training set exactly once.

    The default is "at least 2 of 3". Raise it and you buy label quality with
    dataset size, which is usually the right trade: a smaller, cleaner set
    beats a larger, noisier one at every scale this repo is about.
    """
    if not 0.0 < min_agreement <= 1.0:
        raise ValueError(f"min_agreement must be in (0, 1], got {min_agreement}")

    report = DistillReport()
    for c in candidates:
        _, agreement = majority_vote(c.samples)
        if agreement + _EPS < min_agreement:
            report.dropped.append(
                (c, f"teacher self-agreement {agreement:.2f} < {min_agreement:.2f}")
            )
            continue
        report.kept.append(c)
    return report


def to_examples(candidates: Sequence[Candidate]) -> list[Example]:
    """
    Turn surviving candidates into training examples.

    The majority answer becomes the completion, and `source` records that a
    model wrote this label. That field is not bookkeeping: months later,
    "which of these rows were machine-labelled" is a question somebody will
    ask, and the answer should be a filter rather than an investigation.
    """
    return [
        Example(prompt=c.prompt, completion=majority_vote(c.samples)[0], source=c.source)
        for c in candidates
    ]


# ---------------------------------------------------------------------------
# Measuring the right thing
# ---------------------------------------------------------------------------


def label_accuracy(candidates: Sequence[Candidate]) -> float | None:
    """
    How often the teacher's majority answer matches gold.

    Returns None when no candidate carries a gold label — which is the honest
    answer, and better than a comforting number computed from nothing.
    """
    scored = [c for c in candidates if c.gold is not None]
    if not scored:
        return None
    hits = sum(
        1 for c in scored if normalize(majority_vote(c.samples)[0]) == normalize(c.gold or "")
    )
    return hits / len(scored)


def agreement_rate(student: dict[str, str], teacher: dict[str, str]) -> float:
    """
    Fraction of prompts where the student reproduces the teacher's answer.

    Report it, but never as the headline. This number is FIDELITY, not
    quality: a student that faithfully reproduces a teacher which is wrong
    20% of the time scores beautifully here and is wrong 20% of the time.
    """
    shared = set(student) & set(teacher)
    if not shared:
        return 0.0
    return sum(1 for k in shared if normalize(student[k]) == normalize(teacher[k])) / len(shared)


def accuracy_against_gold(answers: dict[str, str], gold: dict[str, str]) -> float:
    """
    The number that actually matters. Score on a held-out set the teacher
    never labelled, or you are measuring memorisation again.
    """
    shared = set(answers) & set(gold)
    if not shared:
        return 0.0
    return sum(1 for k in shared if normalize(answers[k]) == normalize(gold[k])) / len(shared)


def expected_accuracy(label_acc: float, fidelity: float) -> float:
    """
    A LOWER BOUND on student accuracy: `label_acc * fidelity`.

    The student is right when it reproduces a label that was correct. It can
    also get lucky — disagree with a wrong label and happen to be right — and
    this ignores that, which is why it is a floor rather than an estimate.

    The useful consequence is the one people miss: with perfect fidelity, the
    student is EXACTLY as accurate as its labels. **Distillation cannot
    exceed its teacher on the labels it was given.** If you need better than
    the teacher, the lever is better labels, not more epochs.

    A limitation worth stating rather than hiding: this bound is SYMMETRIC in
    its two arguments, so it cannot tell you whether five points of label
    accuracy beat five points of fidelity — `0.95 * 0.90` and `0.90 * 0.95`
    are the same number. Ranking those two levers needs the real error
    distribution, not this floor. (I asserted otherwise in an early draft;
    a test disagreed, and the test was right.)
    """
    for name, v in (("label_acc", label_acc), ("fidelity", fidelity)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {v}")
    return label_acc * fidelity
