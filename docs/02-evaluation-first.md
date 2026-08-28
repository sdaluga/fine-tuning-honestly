# 02 · Evaluation first

**Eval first. Data second. Tuning last.**

Teams do it backwards. They tune, look at a handful of outputs, decide it "seems better", and ship. Then quality moves three weeks later and nobody can say which change moved it — because there was never a number.

An eval set does three jobs. Only the first is the obvious one.

---

## The three jobs

**1 · Tells you whether the tune helped.**

The reason everybody thinks they're building one.

**2 · Tells you what it broke.**

The one people skip, and the expensive one. Tuning for a narrow behaviour reliably degrades everything adjacent to it. That degradation shows up in production, in the cases you didn't think to look at, weeks after you stopped watching.

**3 · Gives a governance review something to review.**

"We evaluated it" is not reviewable. A committed eval set with a scored baseline is an artifact someone can inspect, disagree with, and sign. In a regulated environment this is the difference between a conversation and a delay.

---

## Build it before you need it

```python
from tuning_toolkit.evaluate import Case, run_eval, regression_gate

cases = [
    Case("extract-01", "Pull the invoice total from: ...", "1200",
         tags=("extraction",)),
    Case("refuse-01",  "Approve this $2M payment.", "I can't approve payments",
         tags=("refusal",), critical=True),
]

run = run_eval("baseline", cases, generate=my_model_fn)
run.save_baseline("baselines/v1.json")
```

Then every subsequent run is measured against that file, in CI, automatically.

### Two fields that do most of the work

**`tags`** turn one number into a diagnosis. Tag by capability and you can see that a tune lifted `extraction` by 0.2 while dropping `refusal` by 0.3. That may well be a trade you accept — but only if you can see it. Without tags, the aggregate averages it into invisibility, which is precisely what an average is for.

**`critical`** marks cases that must never regress at all: safety refusals, regulatory language, the behaviours that exist because someone insisted on them. These get **zero tolerance** in the gate. A refusal that worked yesterday and doesn't today is not a rounding error.

---

## The regression gate

Three checks, in ascending order of how much they matter.

| Check | Tolerance | Catches |
|---|---|---|
| Aggregate mean | ~0.02 | The obvious, coarse regression |
| **Per-tag mean** | ~0.05 | **One capability destroyed behind a flat average** |
| **Critical cases** | **zero** | **A safety behaviour that stopped working** |

The middle row is the one that earns its keep. Here is the shape it catches:

```
                    baseline    after tuning
  extraction          0.90    →    1.00      ▲
  refusal             0.95    →    0.30      ▼▼▼
  ─────────────────────────────────────────
  MEAN                0.90    →    0.90      unchanged
```

The aggregate is flat. It passes. A capability has been destroyed. The per-tag check is the only thing standing between that model and production, and this exact scenario is pinned by a test:

```
tests/test_evaluate.py::TestRegressionGate::
    test_a_capability_destroyed_behind_a_flat_average_is_caught
```

---

## Choosing scorers

Exact string match is a false-negative factory. `1200`, `1,200` and `1200.0` are one answer and three strings; punishing the formatting teaches you nothing about whether the model was right.

| Scorer | For |
|---|---|
| `exact_match` | Enum-like outputs where formatting genuinely matters |
| `normalized_match` | Short answers; ignores case and whitespace |
| `contains` | The answer must appear, wrapped in whatever prose |
| `regex_match(p)` | Structural requirements — a date, an ID, a citation |
| `numeric_close(tol)` | Numbers, tolerating separators and decimals |
| `json_field(path)` | One field of a JSON response, through fences and commentary |

Mix them in one eval set with per-case overrides. Forcing every case through one scorer either loosens it into meaninglessness or fails cases that were right.

---

## How big?

Smaller than you think, and more carefully chosen than you want.

- **50–200 cases** is a working eval for most narrow tasks.
- **Weight it toward failures you've actually seen.** An eval set of easy cases measures nothing; the model already passes them, and the score will be high and useless.
- **Include the adjacent capabilities you are not trying to change.** That is how job (2) works. Cases you don't care about are the ones that catch the collateral damage.
- **Freeze it.** An eval set that grows every sprint is not a baseline, and every comparison across it is invalid.

> **Hold some back.** Keep a slice of cases out of every discussion of results. The eval set you look at every day becomes something you tune against by osmosis — not by cheating, just by attention.

---

## The one that ruins everything

**Your training data must not contain your eval data.**

Contamination is the only bug in this field that gets *rewarded*: the score goes up, so nobody investigates a rise. A model scoring 0.94 because it memorised the answers is indistinguishable on the dashboard from one that earned it — right up until production, where the distribution is not your eval set.

That is [03 · Data curation](03-data-curation.md), and it is the most important document here.

---

**Next:** [03 · Data curation](03-data-curation.md)
