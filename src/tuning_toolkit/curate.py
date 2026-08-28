"""
Data curation — the part that actually determines whether tuning works.
======================================================================

Hyperparameters get the attention. Data gets the result.

A tuning run is a compression of your dataset into weights, faithfully
including its mistakes. Every defect below has been observed to survive
straight through a training run and out the other side into production:

  * Duplicates       over-weight whatever they say, silently
  * Contamination    eval examples leaking into train, so your score is a
                     memorisation measurement wearing a benchmark's clothes
  * PII              becomes extractable from the model, permanently, and
                     "delete the record" is no longer a thing you can do
  * Format drift     one malformed record in a thousand teaches the model
                     that malformed output is sometimes correct
  * Split leakage    near-duplicates split across train and eval, which is
                     contamination that passes an exact-match check

Every function here is pure, deterministic, and runs on a laptop with no
model, no GPU and no API key. That is deliberate: this is the layer you want
covered by tests, because it is the layer whose failures are invisible.

THE RULE THIS MODULE ENFORCES: FAIL LOUD, NOT CLEAN.
Detectors here return findings for a human to act on. They do not quietly
"fix" your data. A silent scrubber that misses one record in ten thousand is
worse than no scrubber, because you will trust it.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """
    Canonical form for comparison ONLY — never for storage.

    Unicode NFKC, lowercase, collapse whitespace, strip. The point is that
    "Hello  world", "hello world" and "Hello world" must collide, because
    to a tokenizer they are near enough identical to over-weight the example,
    and an exact-hash dedup that misses them is theatre.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def content_hash(text: str) -> str:
    """Stable hash of the normalised text. Used for dedup and for manifests."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


#: Punctuation is stripped for SIMILARITY comparison but not for `normalize`,
#: and the asymmetry is deliberate. Exact dedup and the split hash stay
#: conservative — two strings differing by a comma really are two strings, and
#: a hash that ignores that is a hash you cannot reason about. Similarity is
#: the opposite problem: scraped corpora differ by trailing punctuation
#: constantly, and a near-dedup that misses "…river bank" vs "…river bank!"
#: is a near-dedup that misses most of what it was built for.
_PUNCT = re.compile(r"[^\w\s]")


def shingle_normalize(text: str) -> str:
    """Normalisation for similarity comparison: `normalize` plus punctuation."""
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", normalize(text))).strip()


def shingles(text: str, n: int = 5) -> set[str]:
    """
    Word n-grams, for near-duplicate and contamination detection.

    n=5 is the working default: short enough to catch a paraphrase that keeps
    a distinctive clause, long enough that ordinary English doesn't collide.
    Drop to 3 and every document about the same topic looks like a duplicate.
    """
    words = shingle_normalize(text).split()
    if len(words) < n:
        # Short strings get one shingle rather than none, so they can still be
        # compared. Returning an empty set here would make every short record
        # silently un-checkable — a hole big enough to drive an eval through.
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Example:
    """
    One training example, in the chat shape every current tuning API takes.

    `source` is not decoration. When a regulator, a customer or your own
    incident review asks "where did the model learn that", the answer has to
    be a lookup, not an archaeology project.
    """

    prompt: str
    completion: str
    source: str = "unknown"

    @property
    def text(self) -> str:
        return f"{self.prompt}\n{self.completion}"

    @property
    def id(self) -> str:
        return content_hash(self.text)[:16]


@dataclass
class Finding:
    """Something a human needs to look at. Never auto-resolved."""

    kind: str
    example_id: str
    detail: str
    severity: str = "warn"  # "warn" | "block"


@dataclass
class CurationReport:
    kept: list[Example] = field(default_factory=list)
    dropped: list[tuple[Example, str]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "block"]

    @property
    def is_releasable(self) -> bool:
        """
        A dataset with blocking findings does not get trained on. Not "gets
        trained on with a note in the ticket" — the gate is the point.
        """
        return not self.blocking

    def summary(self) -> dict[str, int]:
        return {
            "kept": len(self.kept),
            "dropped": len(self.dropped),
            "findings": len(self.findings),
            "blocking": len(self.blocking),
        }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def exact_dedup(examples: Sequence[Example]) -> tuple[list[Example], list[tuple[Example, str]]]:
    """
    Collapse byte-identical-after-normalisation examples, keeping the first.

    Cheap, and catches most of it. Scraped and generated datasets routinely
    arrive 10-30% duplicated, and every duplicate is a silent thumb on the
    scale for whatever that example teaches.
    """
    seen: dict[str, Example] = {}
    kept: list[Example] = []
    dropped: list[tuple[Example, str]] = []

    for ex in examples:
        h = content_hash(ex.text)
        if h in seen:
            dropped.append((ex, f"exact duplicate of {seen[h].id}"))
            continue
        seen[h] = ex
        kept.append(ex)

    return kept, dropped


def near_dedup(
    examples: Sequence[Example], threshold: float = 0.8, n: int = 5
) -> tuple[list[Example], list[tuple[Example, str]]]:
    """
    Collapse near-duplicates by n-gram Jaccard similarity.

    O(n^2), and honestly so: at the scale where that hurts you want MinHash
    LSH, and the right move is to swap the implementation rather than to
    pretend the naive one scales. Correctness first; the interface does not
    change when you make it fast.
    """
    kept: list[Example] = []
    dropped: list[tuple[Example, str]] = []
    kept_shingles: list[tuple[Example, set[str]]] = []

    for ex in examples:
        sh = shingles(ex.text, n)
        hit = next(
            ((other, sim) for other, osh in kept_shingles
             if (sim := jaccard(sh, osh)) >= threshold),
            None,
        )
        if hit:
            other, sim = hit
            dropped.append((ex, f"near-duplicate of {other.id} (jaccard {sim:.2f})"))
            continue
        kept.append(ex)
        kept_shingles.append((ex, sh))

    return kept, dropped


# ---------------------------------------------------------------------------
# Decontamination
# ---------------------------------------------------------------------------


def decontaminate(
    train: Sequence[Example], eval_set: Sequence[Example], threshold: float = 0.5, n: int = 5
) -> tuple[list[Example], list[tuple[Example, str]]]:
    """
    Remove training examples that overlap the evaluation set.

    THE MOST IMPORTANT FUNCTION IN THIS FILE.

    Skip this and your eval score measures memorisation. The failure is
    perfectly silent and it points the wrong way: the number goes UP, so
    nobody investigates. A model that scores 0.94 because it saw the answers
    is indistinguishable, on the dashboard, from one that earned it — right
    up until production, where the distribution is not your eval set.

    The threshold is lower than `near_dedup`'s on purpose. For duplicates you
    want confidence before discarding data; for contamination you want
    suspicion, because a false positive costs you one training row and a
    false negative costs you the credibility of every number you report.
    """
    eval_shingles = [shingles(e.text, n) for e in eval_set]
    # Prompt-only shingles too: an eval item leaks just as thoroughly when the
    # training row shares its question and differs in the answer.
    eval_prompt_shingles = [shingles(e.prompt, n) for e in eval_set]

    kept: list[Example] = []
    dropped: list[tuple[Example, str]] = []

    for ex in train:
        full = shingles(ex.text, n)
        prompt_only = shingles(ex.prompt, n)
        worst = 0.0
        for esh, epsh in zip(eval_shingles, eval_prompt_shingles):
            worst = max(worst, jaccard(full, esh), jaccard(prompt_only, epsh))
            if worst >= threshold:
                break
        if worst >= threshold:
            dropped.append((ex, f"contaminates eval set (overlap {worst:.2f})"))
            continue
        kept.append(ex)

    return kept, dropped


# ---------------------------------------------------------------------------
# PII detection
# ---------------------------------------------------------------------------

#: First-pass detectors. NOT a compliance control and not represented as one.
#: They catch the shapes that leak most often through a copy-paste pipeline.
#: Names, addresses and account references need a real NER or DLP pass —
#: Microsoft Purview, AWS Macie, Google DLP — and this is where you hand off.
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "us_phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card_like": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "aws_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def detect_pii(text: str) -> list[str]:
    """Return the names of every detector that fired. Order is stable."""
    return [name for name, pat in PII_PATTERNS.items() if pat.search(text)]


def scan_pii(examples: Iterable[Example]) -> list[Finding]:
    """
    Flag examples containing probable PII or secrets.

    Severity is BLOCK, always, and the finding is never auto-resolved.

    The asymmetry justifies it. Training on PII is not a bug you patch later:
    the data is in the weights, it has been shown to be extractable, and no
    deletion request can reach it. The only cheap moment is this one.
    """
    findings: list[Finding] = []
    for ex in examples:
        for kind in detect_pii(ex.text):
            findings.append(
                Finding(
                    kind=f"pii:{kind}",
                    example_id=ex.id,
                    detail=f"{kind} pattern found in example from {ex.source!r}",
                    severity="block",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Format validation
# ---------------------------------------------------------------------------


def validate_format(examples: Iterable[Example], max_chars: int = 32_000) -> list[Finding]:
    """
    Catch the malformed records that teach malformed behaviour.

    An empty completion is the quiet killer: a handful of them teach the model
    that silence is an acceptable response, and it will take that lesson to
    production and apply it to your most important question.
    """
    findings: list[Finding] = []
    for ex in examples:
        if not ex.prompt.strip():
            findings.append(Finding("format:empty_prompt", ex.id, "prompt is empty", "block"))
        if not ex.completion.strip():
            findings.append(
                Finding(
                    "format:empty_completion",
                    ex.id,
                    "completion is empty — teaches the model that silence is valid",
                    "block",
                )
            )
        if len(ex.text) > max_chars:
            findings.append(
                Finding(
                    "format:too_long",
                    ex.id,
                    f"{len(ex.text)} chars exceeds {max_chars}; will be truncated mid-example",
                    "warn",
                )
            )
        if ex.source == "unknown":
            findings.append(
                Finding(
                    "lineage:no_source",
                    ex.id,
                    "no source recorded — this row cannot be traced or retracted",
                    "warn",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split(
    examples: Sequence[Example], eval_fraction: float = 0.1
) -> tuple[list[Example], list[Example]]:
    """
    Deterministic content-hash split.

    Hash-based rather than random, for two reasons that both bite later:
    the split is reproducible across machines and runs without carrying a
    seed around, and identical content always lands on the same side — so a
    duplicate that survived dedup cannot straddle the split and quietly
    contaminate the eval.
    """
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}")

    train: list[Example] = []
    held: list[Example] = []
    cutoff = int(eval_fraction * 0xFFFF)

    for ex in examples:
        bucket = int(content_hash(ex.text)[:4], 16)
        (held if bucket < cutoff else train).append(ex)

    return train, held


def assert_no_leakage(train: Sequence[Example], held: Sequence[Example]) -> list[Finding]:
    """
    Verify after the fact that nothing appears on both sides.

    A belt-and-braces check over `split`. It exists because this invariant is
    load-bearing for every number the project will publish, and an assertion
    you can run is worth more than an argument that the code is correct.
    """
    train_hashes = {content_hash(e.text) for e in train}
    return [
        Finding(
            "split:leakage",
            e.id,
            "example appears in both train and eval — every eval number is invalid",
            "block",
        )
        for e in held
        if content_hash(e.text) in train_hashes
    ]


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def curate(
    examples: Sequence[Example],
    eval_set: Sequence[Example] | None = None,
    *,
    near_dup_threshold: float = 0.8,
    contamination_threshold: float = 0.5,
) -> CurationReport:
    """
    Run the whole pipeline in the order that matters.

    Order is not cosmetic:

      1. validate  — before spending effort on records that are broken
      2. exact     — cheapest reduction first
      3. near      — expensive, so run it on the already-reduced set
      4. decontam  — LAST of the removals, so it sees the final training set
      5. pii       — on what actually survives, so no finding is about a row
                     that was going to be dropped anyway

    Running decontamination before dedup would let a duplicate of a
    contaminating record slip back in behind it.
    """
    report = CurationReport()

    report.findings.extend(validate_format(examples))

    kept, dropped = exact_dedup(examples)
    report.dropped.extend(dropped)

    kept, dropped = near_dedup(kept, threshold=near_dup_threshold)
    report.dropped.extend(dropped)

    if eval_set:
        kept, dropped = decontaminate(kept, eval_set, threshold=contamination_threshold)
        report.dropped.extend(dropped)

    report.findings.extend(scan_pii(kept))
    report.kept = kept
    return report
