"""
The artifacts a review board asks for — generated, not written by hand.
======================================================================

In a regulated industry, a tuned model is not a build output. It is a
controlled asset, and somebody will eventually ask three questions about it:

    1. What data went into this model?
    2. Who approved it, against what evidence?
    3. Can you reproduce it?

Answering those from memory, months later, is how a review turns into an
archaeology project. Answering them from a file committed alongside the run
takes a minute, and the file costs nothing to produce IF it is generated at
training time rather than reconstructed at audit time.

That timing is the whole idea. A model card written after the fact is a
document. A model card generated from the actual dataset, with the actual
content hashes, at the moment of the actual run, is EVIDENCE.

WHAT THIS IS NOT
----------------
Not compliance advice, and not a substitute for your own control framework.
It produces the substrate those frameworks need: an immutable record of what
went in, what came out, and what was checked in between. Map it to whichever
regime applies to you.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

from .curate import CurationReport, Example, content_hash


# ---------------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetManifest:
    """
    An immutable record of exactly what a training run consumed.

    `fingerprint` is the load-bearing field. It is derived from the sorted
    content hashes of every example, so two runs that claim the same dataset
    can be PROVEN to have used the same dataset — or shown not to have. That
    turns "we think this was trained on v3" into a comparison.
    """

    name: str
    version: str
    created_utc: str
    example_count: int
    fingerprint: str
    sources: dict[str, int]
    checks_run: list[str]
    findings_summary: dict[str, int]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)


def build_manifest(
    name: str,
    version: str,
    report: CurationReport,
    checks_run: Sequence[str],
    *,
    now: datetime | None = None,
) -> DatasetManifest:
    """
    Build a manifest from a completed curation run.

    Takes the CurationReport rather than a bare list, so the manifest records
    what was CHECKED as well as what was kept. A dataset of 10,000 clean rows
    means nothing on its own; 10,000 rows that passed decontamination and a
    PII scan is a claim someone can act on.
    """
    fingerprint = fingerprint_examples(report.kept)

    sources: dict[str, int] = {}
    for ex in report.kept:
        sources[ex.source] = sources.get(ex.source, 0) + 1

    kinds: dict[str, int] = {}
    for f in report.findings:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1

    return DatasetManifest(
        name=name,
        version=version,
        created_utc=(now or datetime.now(timezone.utc)).isoformat(),
        example_count=len(report.kept),
        fingerprint=fingerprint,
        sources=dict(sorted(sources.items())),
        checks_run=sorted(checks_run),
        findings_summary=dict(sorted(kinds.items())),
    )


def fingerprint_examples(examples: Sequence[Example]) -> str:
    """
    Order-independent fingerprint of a dataset.

    Sorted before hashing, deliberately: shuffling the training set must not
    change its identity, or every reshuffle looks like a different dataset
    and the field becomes noise nobody trusts.
    """
    hashes = sorted(content_hash(e.text) for e in examples)
    return hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------


@dataclass
class ModelCard:
    """
    The reviewable summary of one tuned model.

    Fields chosen by what gets asked in an actual architecture and
    responsible-AI review, not by what is pleasant to write. The awkward
    ones — `known_limitations`, `out_of_scope_uses` — are the ones that make
    the card worth reading. A card with an empty limitations section tells a
    reviewer that nobody looked, and they will be right.
    """

    model_name: str
    base_model: str
    version: str
    owner: str
    intended_use: str
    method: str  # "lora" | "qlora" | "sft" | "dpo" | "distillation" | ...
    dataset: DatasetManifest
    eval_summary: dict[str, float] = field(default_factory=dict)
    known_limitations: list[str] = field(default_factory=list)
    out_of_scope_uses: list[str] = field(default_factory=list)
    approvals: list[str] = field(default_factory=list)
    hyperparameters: dict[str, object] = field(default_factory=dict)
    created_utc: str = ""

    def __post_init__(self) -> None:
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).isoformat()

    def gaps(self) -> list[str]:
        """
        What a reviewer will send this back for.

        Run it in CI and the card cannot rot: the build fails before anyone's
        calendar is involved, which is a cheaper place to find the gap than a
        review board meeting.
        """
        out: list[str] = []
        if not self.eval_summary:
            out.append("no eval results — the model's quality is unmeasured")
        if not self.known_limitations:
            out.append(
                "no known limitations — every model has them; an empty list "
                "means nobody looked"
            )
        if not self.out_of_scope_uses:
            out.append(
                "no out-of-scope uses — without these the card licenses "
                "everything by omission"
            )
        if not self.approvals:
            out.append("no recorded approval")
        if not self.dataset.checks_run:
            out.append("dataset manifest records no checks")
        return out

    @property
    def is_reviewable(self) -> bool:
        return not self.gaps()

    def to_markdown(self) -> str:
        d = self.dataset
        lines = [
            f"# Model card — {self.model_name}",
            "",
            f"**Version** `{self.version}`  ·  **Base model** `{self.base_model}`  "
            f"·  **Method** `{self.method}`",
            f"**Owner** {self.owner}  ·  **Created** {self.created_utc}",
            "",
            "## Intended use",
            "",
            self.intended_use,
            "",
            "## Training data",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Dataset | `{d.name}` v`{d.version}` |",
            f"| Examples | {d.example_count:,} |",
            f"| Fingerprint | `{d.fingerprint[:16]}…` |",
            f"| Checks run | {', '.join(d.checks_run) or '**none**'} |",
            "",
            "**Sources**",
            "",
        ]
        lines += [f"- `{src}` — {n:,} examples" for src, n in d.sources.items()] or ["- none recorded"]

        if d.findings_summary:
            lines += ["", "**Curation findings**", ""]
            lines += [f"- `{k}` — {n}" for k, n in d.findings_summary.items()]

        lines += ["", "## Evaluation", ""]
        if self.eval_summary:
            lines += ["| Metric | Score |", "|---|---|"]
            lines += [f"| {k} | {v:.3f} |" for k, v in sorted(self.eval_summary.items())]
        else:
            lines.append("**Not evaluated.** This model is not releasable.")

        if self.hyperparameters:
            lines += ["", "## Hyperparameters", "", "```json",
                      json.dumps(self.hyperparameters, indent=2, sort_keys=True), "```"]

        lines += ["", "## Known limitations", ""]
        lines += [f"- {x}" for x in self.known_limitations] or ["- **None recorded — this is a gap.**"]

        lines += ["", "## Out of scope", ""]
        lines += [f"- {x}" for x in self.out_of_scope_uses] or ["- **None recorded — this is a gap.**"]

        lines += ["", "## Approvals", ""]
        lines += [f"- {x}" for x in self.approvals] or ["- **None recorded — this is a gap.**"]

        if gaps := self.gaps():
            lines += ["", "## ⚠ Outstanding gaps", ""]
            lines += [f"- {g}" for g in gaps]

        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def reproducibility_check(card: ModelCard, examples: Sequence[Example]) -> list[str]:
    """
    Verify that a dataset on disk today is the one this card describes.

    The question this answers is "is what I'm holding the thing that was
    approved?" — and it is asked most often at exactly the moment when
    everyone's memory has become unreliable, six months after the run, during
    an incident.
    """
    problems: list[str] = []

    actual = fingerprint_examples(examples)
    if actual != card.dataset.fingerprint:
        problems.append(
            f"dataset fingerprint mismatch: card says {card.dataset.fingerprint[:16]}…, "
            f"data on disk is {actual[:16]}… — this is NOT the approved dataset"
        )

    if len(examples) != card.dataset.example_count:
        problems.append(
            f"example count mismatch: card says {card.dataset.example_count}, "
            f"found {len(examples)}"
        )

    return problems
