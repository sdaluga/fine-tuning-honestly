# 03 · Data curation

Hyperparameters get the attention. Data gets the result.

A tuning run is a compression of your dataset into weights, faithfully including its mistakes. Every defect below has been observed to survive a training run intact and arrive in production.

None of them raise an exception. One of them makes your metrics look *better*.

---

## The five defects

| Defect | What it does | Detection |
|---|---|---|
| **Duplicates** | Silently multiply one example's influence | `exact_dedup`, `near_dedup` |
| **Contamination** | Your eval score becomes a memorisation measurement | `decontaminate` |
| **PII** | Becomes extractable from the model, permanently | `scan_pii` |
| **Format drift** | Teaches the model that malformed output is sometimes right | `validate_format` |
| **Split leakage** | Contamination that passes an exact-match check | `split`, `assert_no_leakage` |

Run all five:

```bash
python examples/02-curate-a-dataset/curate_dataset.py
```

Fourteen plausible-looking operational examples, seeded with every one of these. The run finds all of them and refuses to release the dataset.

---

## 1 · Duplicates

Scraped and generated datasets routinely arrive 10–30% duplicated. Every duplicate is a thumb on the scale for whatever that example teaches — and you never chose to press it.

Exact dedup runs on **normalised** text (NFKC, lowercased, whitespace collapsed), because `"Hello  World"` and `"hello world"` are one example to a tokenizer and two strings to a naive hash.

Near-dedup uses word 5-gram Jaccard. There is one asymmetry worth knowing:

> **Punctuation is stripped for similarity, but not for hashing.** Exact dedup and the split hash stay conservative — two strings differing by a comma really are two strings. Similarity is the opposite problem: corpora differ by trailing punctuation constantly, and a near-dedup that misses `…river bank` vs `…river bank!` misses most of what it was built for.

---

## 2 · Contamination — the important one

**Training examples that overlap your evaluation set.**

Skip this and your eval measures memorisation. The failure is perfectly silent and it points the wrong way: **the number goes up**, so nobody investigates.

The threshold for contamination is deliberately *lower* than for deduplication:

| | Threshold | Reasoning |
|---|---|---|
| Near-dedup | 0.8 | Want confidence before discarding training data |
| **Decontamination** | **0.5** | Want *suspicion*. A false positive costs one row. A false negative costs the credibility of every number you report |

Contamination is also checked **prompt-to-prompt**, not just on full text. A training row carrying the eval's question with a *different* answer still teaches the model that exact question — and a full-text comparison misses it entirely once the answers diverge enough. That path has its own test, and removing it turns the test red:

```
tests/test_curate.py::TestDecontamination::
    test_sharing_only_the_question_still_counts_as_contamination
```

---

## 3 · PII — the one you cannot undo

Training on PII is not a defect you patch in the next release. **The data is in the weights, it has been shown to be extractable, and no deletion request can reach it.** Curation is the last cheap moment.

That asymmetry is why PII findings are `severity="block"` and never auto-resolved.

The detectors here — email, phone, SSN-shaped, card-shaped, IPv4, cloud keys, private key headers — are a **first pass, not a compliance control**, and this repo does not pretend otherwise. They catch the shapes that leak through a copy-paste pipeline. Names, addresses and account references need a real NER or DLP pass: Microsoft Purview, AWS Macie, Google DLP. This layer is where you hand off, and knowing where the handoff is beats a tool that claims it doesn't need one.

---

## 4 · Format

The quiet killer is the **empty completion**. A handful of them teach the model that silence is an acceptable response, and it will apply that lesson in production to your most important question.

Also flagged: overlong examples that will be truncated mid-thought, and — as a warning — **any row with no recorded source**. A row you cannot trace is a row you cannot retract when someone asks you to.

---

## 5 · Splitting

The split is **content-hash based, not random**. Two reasons, both of which bite later:

- **Reproducible** across machines and runs with no seed to carry around.
- **Identical content always lands on the same side.** A duplicate that survived dedup cannot straddle the split and quietly contaminate the eval.

`assert_no_leakage` then verifies the invariant after the fact. That is belt and braces on purpose: this invariant is load-bearing for every number the project will publish, and an assertion you can run beats an argument that the code is correct.

---

## Order matters

```
1. validate    →  before spending effort on broken records
2. exact dedup →  cheapest reduction first
3. near dedup  →  expensive, so run it on the reduced set
4. decontaminate → LAST removal, so it sees the final training set
5. PII scan    →  on what survives, so no finding is about a dropped row
```

Run decontamination before dedup and a duplicate of a contaminating record slips back in behind it. That ordering has its own test.

---

## Fail loud, not clean

Every detector here **reports**. None of them quietly fix your data.

A silent scrubber that misses one row in ten thousand is worse than no scrubber, because you will trust it. So `curate()` returns findings and a gate:

```python
report = curate(train, eval_set)
if not report.is_releasable:
    raise SystemExit("blocking findings — fix the data")
```

Removing flagged rows is a **human** step, and the example shows it as one. That is not ceremony: it puts a person's judgement between a detector's guess and a dataset that is about to become weights.

---

> **Machine-generated data is not exempt.** Teacher output from a distillation run duplicates, contaminates and leaks PII exactly like human-written data — more so, because it was produced in bulk. Run it through the same pipeline, and tag its `source` so "which rows were machine-labelled" stays a filter rather than an investigation. See [`examples/05-distill/`](../examples/05-distill/).

---

**Next:** [04 · Tuning methods](04-tuning-methods.md)
