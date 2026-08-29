# Example 05 — Distil a teacher, without inheriting its bugs

```bash
python examples/05-distill/distill_a_teacher.py
```

No GPU. No API key. Under a second.

Distillation is the most defensible reason to change weights — quality is already fine, cost is not — and it has a failure mode the other methods don't. That failure is social before it is technical:

> **Teacher output gets treated as ground truth because it came from the expensive model.**

It isn't. A teacher that is 80% accurate hands you a label set that is 80% accurate, and the student learns the other 20% as *systematic* error, because the teacher is wrong in consistent ways.

## 1 · The cheapest quality lever there is

Sample the teacher three times. Keep only what it agrees with itself on.

```
dropped [0.33]  elevated bearing temperature on the main feed pump   (wrong too)
dropped [0.33]  intermittent SCADA comms dropout on unit 1           (wrong too)
dropped [0.33]  audit finding response due end of quarter            (wrong too)

kept 19 of 22
label accuracy   0.800  ->  0.941   (+0.141)
```

**Fourteen points of label quality for one extra inference pass** over the training set — far cheaper than the tuning run it protects. Self-consistency isn't a correctness oracle (a model can be confidently and consistently wrong) but it correlates strongly enough to be the first thing you do.

Note that it also discards *correct* labels. Smaller and cleaner beats larger and noisier.

## 2 · Machine-generated does not mean clean

The surviving labels go through the same `curate()` pipeline as human-written data:

```
19 teacher-labelled examples -> curate()
kept 18 · dropped 1 · blocking 1
  dropped: exact duplicate of d0a2221722b794cc
  BLOCK [pii:us_phone] us_phone pattern found in example from 'teacher:big-model-v2'
```

Teacher output duplicates, contaminates and leaks PII exactly like anything else — **more so, because it was produced in bulk.** Every row carries `source="teacher:big-model-v2"`, so "which rows were machine-labelled" is a filter rather than an investigation.

## 3 · The agreement trap

The natural metric for distillation is student-teacher agreement. It is the wrong headline.

```
student agrees with teacher   0.900   <- the flattering number
teacher accuracy vs gold      0.800
STUDENT ACCURACY vs gold      0.700   <- the real one
```

The student reproduces its teacher almost perfectly. That is *what distillation is for* — and it is evidence the copy worked, not evidence the model is good. Agreement measures **fidelity**. Only gold measures quality, and only on a held-out set the teacher never labelled.

If you have no such set, you don't know whether distillation worked. You only know it copied.

## 4 · The ceiling

```
with raw labels        floor ≈ 0.800 × 0.900 = 0.720
with filtered labels   floor ≈ 0.941 × 0.900 = 0.847
```

A student that perfectly reproduces its labels is **exactly as accurate as they are**. Distillation cannot exceed its teacher on the labels it was given — so when you need better than the teacher, the lever is better labels, not more epochs.

That product is a *lower* bound: it ignores the student getting lucky by disagreeing with a wrong label. It is also symmetric in its two terms, so it cannot tell you which lever to pull first — ranking those needs the real error distribution, and the module says so rather than pretending otherwise.

## What's in here

| File | |
|---|---|
| `teacher_samples.jsonl` | 22 prompts × 3 teacher samples. 20 carry gold labels so the example can measure what filtering bought |
| `heldout.json` | A slice the teacher never labelled — gold, teacher answers, student answers |

In a real run you would have no gold on the training set. That is the whole reason you're distilling, and exactly why the held-out set matters.

## It checks itself

The script asserts its own premise before exiting: that filtering improved label accuracy, that agreement really does exceed accuracy, and that curation found something in the teacher output. It exits non-zero otherwise — which it did on first run, because the fixture had no dirty rows to find. The fixture was wrong, not the claim.

**Read next:** [docs/04-tuning-methods.md](../../docs/04-tuning-methods.md)
