# 05 · Governance — what a review board actually asks

In a regulated industry, a tuned model is not a build output. It is a **controlled asset**, and somebody will eventually ask three questions about it.

None of them are hard to answer *if* you generated the answer at training time. All of them turn into an archaeology project if you didn't.

---

## The three questions

### 1 · What data went into this model?

Not "a curated set of internal documents." Which documents, how many, from which systems, with what checks run against them.

```json
{
  "name": "ops-triage-sft",
  "version": "0.1.0",
  "example_count": 7,
  "fingerprint": "c528b208324a015e…",
  "sources": { "engineering-guides": 1, "maint-notes-2026": 1, "triage-labelled": 2 },
  "checks_run": ["decontamination", "exact-dedup", "format", "near-dedup", "pii-scan"],
  "findings_summary": { "lineage:no_source": 0 }
}
```

`checks_run` is the field that makes this reviewable. Seven clean rows means nothing on its own. Seven rows that **passed decontamination and a PII scan** is a claim someone can act on.

`fingerprint` is the load-bearing one — see below.

### 2 · Who approved it, against what evidence?

A model card, generated from the run rather than written afterwards.

The awkward fields are the ones that make it worth reading:

- **`known_limitations`** — every model has them. An empty list tells a reviewer nobody looked, and they are right.
- **`out_of_scope_uses`** — without these, the card licenses everything by omission.
- **`approvals`** — who signed, when, against which version.

`ModelCard.gaps()` returns exactly what a reviewer would send it back for. **Run it in CI** and the card cannot rot: the build fails before anyone's calendar is involved, which is a much cheaper place to find the gap than a review board meeting.

### 3 · Can you reproduce it?

```python
problems = reproducibility_check(card, examples_on_disk)
```

The fingerprint is the sorted SHA-256 of every example's content hash. Sorted, deliberately: **reshuffling training data must not change its identity**, or every reshuffle reads as tampering and the field becomes noise nobody trusts. Editing one row *does* change it.

That turns "we think this was trained on v3" into a comparison you can run:

```
dataset fingerprint mismatch: card says c528b208324a015e…,
data on disk is c3847c9a00f78fb5… — this is NOT the approved dataset
```

This question gets asked six months later, during an incident, when everyone's memory has become unreliable. That is exactly when a hash is worth more than a recollection.

---

## Generated, not written

A model card written after the fact is a **document**. A model card generated from the actual dataset, with the actual content hashes, at the moment of the actual run, is **evidence**.

That timing is the entire idea, and it is why the manifest is written **before** training starts in `train_lora.py` — so an interrupted run still leaves a record of exactly what it was going to consume.

---

## Where this fits an existing control framework

This repo produces the **substrate**, not the framework. It does not tell you which controls apply to you; it makes the artifacts those controls need cheap enough that nobody skips them.

| Control question | Artifact |
|---|---|
| Data lineage and provenance | `DatasetManifest.sources`, per-row `Example.source` |
| Data minimisation / PII handling | `scan_pii` findings, blocking release gate |
| Change control and approval | `ModelCard.approvals`, `gaps()` in CI |
| Reproducibility and integrity | `fingerprint`, `reproducibility_check` |
| Performance monitoring | Committed eval baselines, `regression_gate` |
| Intended use and limitations | `intended_use`, `known_limitations`, `out_of_scope_uses` |

Map these onto whichever regime governs you. The artifacts are the same either way; only the paperwork around them differs.

---

## Three things that bite in regulated environments

**Right-to-erasure does not reach model weights.** Once PII is trained in, it is extractable and permanent, and a deletion request cannot touch it. This is not a policy problem you resolve later — it is a curation gate you enforce now. It is why PII findings **block** in this toolkit rather than warn.

**Your base model will be deprecated.** Your tune was against a specific version. When that version retires you re-tune, re-evaluate and re-approve. That clock starts the day you ship, and it belongs in the operating cost — see the ownership overhead line in [01](01-when-to-tune.md).

**"We evaluated it" is not evidence.** A committed eval set with a scored, versioned baseline is. The difference between those two sentences is usually several weeks of review.

---

## What this is not

Not compliance advice, and not a substitute for your own control framework or your legal team. It is the engineering layer underneath them: an immutable record of what went in, what came out, and what was checked in between — produced automatically, at the only moment when producing it is cheap.

---

**Back to:** [README](../README.md) · **Start at:** [01 · When to tune](01-when-to-tune.md)
