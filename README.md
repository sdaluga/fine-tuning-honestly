<div align="center">

# Fine-Tuning, Honestly

### A working toolkit for the parts of a model-tuning project that must be correct whether or not the training run does — and a straight answer to the question most teams skip: *should you be tuning at all?*

[![License: MIT](https://img.shields.io/badge/License-MIT-2EA043?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-130%20·%20mutation--checked-8957E5?style=for-the-badge)](tests/)
[![No GPU required](https://img.shields.io/badge/toolkit-no%20GPU%20required-FF6F00?style=for-the-badge)](src/tuning_toolkit/)

**[When to tune](docs/01-when-to-tune.md) · [Evaluation first](docs/02-evaluation-first.md) · [Data curation](docs/03-data-curation.md) · [Methods](docs/04-tuning-methods.md) · [Governance](docs/05-governance.md)**

</div>

---

## Why this exists

Fine-tuning is the most requested and least justified step in applied AI. It gets proposed because it sounds like the serious answer — "we wrote a better prompt" survives a steering-committee slide badly, "we trained our own model" survives it well.

Most tutorials hand you a training loop. The training loop is close to a solved problem. What actually decides whether a tuning project works is everything around it: **was this the right move, is the data clean, did anything regress, and can you prove what went in.** Those fail silently, months later, in front of someone who is not on your team.

So the ratio here is deliberate:

| | Lines | Needs a GPU | Tested |
|---|---|---|---|
| The training script | 344 | yes | config validation only |
| Everything that has to be right anyway | ~1,500 | **no** | **130 tests, mutation-checked** |

---

## Quickstart

```bash
git clone https://github.com/sdaluga/fine-tuning-honestly.git
cd fine-tuning-honestly
pip install -e ".[dev]"

python examples/01-decide/decide.py               # should you tune at all?
python examples/02-curate-a-dataset/curate_dataset.py   # find what's wrong with the data
pytest                                             # 130 tests, ~0.2s
```

**No GPU. No API key. No network.** The whole toolkit runs on a laptop, and that is the point: the layers that fail silently are exactly the layers that should be cheap to test.

---

## The ladder

Each rung costs roughly an order of magnitude more than the one below, in engineering time far more than GPU spend. Climb one at a time.

```mermaid
flowchart LR
    P["0 · PROMPT"] --> F["1 · FEW-SHOT"] --> R["2 · RETRIEVAL"] --> T["3 · TOOLS"]
    T --> D["4 · DISTILL"] --> S["5 · SFT"] --> Q["6 · PREFERENCE"] --> C["7 · CONTINUED PT"]

    style P fill:#0d2818,stroke:#238636,stroke-width:2px
    style F fill:#0d2818,stroke:#238636,stroke-width:2px
    style R fill:#0d2818,stroke:#238636,stroke-width:2px
    style T fill:#0d2818,stroke:#238636,stroke-width:2px
    style D fill:#3d1d00,stroke:#bf8700,stroke-width:2px
    style S fill:#3d1d00,stroke:#bf8700,stroke-width:2px
    style Q fill:#4d0f0f,stroke:#da3633,stroke-width:2px
    style C fill:#4d0f0f,stroke:#da3633,stroke-width:2px
```

**Green rungs are reversible.** Amber and red change weights — and that is where an engineering decision becomes a governance one, with a lineage record, an approval, and a place in the model inventory. Nobody mentions that at the start.

```python
from tuning_toolkit import Scenario, recommend

recommend(Scenario(failure_is_missing_knowledge=True)).rung
# Rung.RETRIEVAL — "Fine-tuning would freeze a moving corpus into weights,
#                   with no citation path and no way to correct one fact
#                   without a retrain."
```

---

## The three examples

### 🧭 [01 · Should we tune?](examples/01-decide/) — no GPU

Four real scenarios. Three of them sound like tuning problems in a planning meeting. One is.

Also prices the decision — including **the line most business cases omit**: the monthly cost of *owning* a tuned model. Same tuning, same per-token advantage, two volumes:

| | 4M req/month | 25k req/month |
|---|---|---|
| Monthly saving | **$15,100** | **−$3,384** |
| **Breakeven** | **3.0 months** | **never** |

Tuning didn't fail in the right-hand column. It was never going to work, and thirty seconds of arithmetic says so before anyone spends a quarter finding out.

### 🧹 [02 · Curate a dataset](examples/02-curate-a-dataset/) — no GPU

Fourteen plausible-looking operational examples, seeded with every defect this toolkit exists to catch: an exact duplicate, two near-duplicates, two rows that contaminate the eval set, one row carrying an email address and a phone number, one empty completion.

None of them announce themselves. Every one of them would train. The run finds all five, **refuses to release the dataset**, and emits a dataset manifest and a model card that lists its own gaps.

### 🔧 [03 · The actual LoRA fine-tune](examples/03-lora-finetune/) — needs a GPU

Real `peft` + `trl` code, three configs, and two hard stops: it **will not train on an uncurated dataset**, and it **will not train without a baseline eval**. Not flags you can pass.

```bash
python examples/03-lora-finetune/train_lora.py --config configs/lora-7b.yaml --validate-only
# config OK — rank 16, alpha 32, lr 0.0001, 3 epochs
# REFUSING TO TRAIN — the dataset has blocking findings:
#   [pii:email] email pattern found in example from 'contact-sheet'
```

---

## Ten things that surprise people

1. **Most "fine-tuning problems" are retrieval problems.** The model doesn't lack capability, it lacks the document.
2. **Distillation is the most defensible reason to change weights** and the least discussed.
3. **Data quality beats hyperparameters, and it isn't close.** Rank 8 on curated data beats rank 64 on scraped data.
4. **A duplicated example silently doubles its own influence.** Scraped sets arrive 10–30% duplicated.
5. **Eval contamination makes your score go _up_** — the only bug in this field that gets rewarded.
6. **Tuning one behaviour degrades adjacent ones,** and the aggregate hides it by construction.
7. **PII in training data cannot be undone.** It's in the weights, it's extractable, and no deletion request reaches it.
8. **Three epochs is usually the ceiling.** Beyond that you're memorising and the eval is too small to say so.
9. **Owning a tuned model can cost more than the tokens it saves.**
10. **The base model moves under you.** Your tune was against a version, and that clock starts the day you ship.

---

## The tests are the argument

```bash
pytest          # 130 tests, ~0.2 seconds, no GPU, no API key
```

Every test is named for the production incident it prevents, because a test called `test_decontaminate_works` tells a future reader nothing about whether it is safe to delete.

**All thirteen controls were mutation-checked** — deliberately broken, one at a time, to confirm the suite goes red:

| Control broken | Result |
|---|---|
| Decontamination made a no-op | 4 tests fail |
| Prompt-only overlap check removed | 1 test fails |
| PII findings downgraded to warnings | 2 tests fail |
| Split switched to random sampling | 3 tests fail |
| Leakage assertion always returns clean | 1 test fails |
| Empty-completion check downgraded | 1 test fails |
| Decision ladder reordered | 1 test fails |
| Eval-set precondition removed | 1 test fails |
| Cost model ignores ownership overhead | 5 tests fail |
| Regression gate drops per-tag check | 1 test fails |
| Critical cases lose zero-tolerance | 1 test fails |
| Fingerprint stops sorting | 2 tests fail |
| Model card gap check silenced | 7 tests fail |

Two of those initially survived, and both were bugs in the *tests*, not the code — the honest outcome of doing this rather than claiming it. Both are documented in the test files that fixed them.

---

## Repo layout

```
src/tuning_toolkit/
├── decision.py      the ladder, the blockers, the cost model      pure · 27 tests
├── curate.py        dedup · decontamination · PII · split         pure · 47 tests
├── evaluate.py      eval harness + regression gate                pure · 31 tests
└── governance.py    manifests · model cards · reproducibility     pure · 25 tests

examples/
├── 01-decide/                 four scenarios, four answers        no GPU
├── 02-curate-a-dataset/       find five planted defects           no GPU
└── 03-lora-finetune/          real peft + trl, three configs      GPU

docs/
├── 01-when-to-tune.md         the ladder and the economics
├── 02-evaluation-first.md     eval first, data second, tuning last
├── 03-data-curation.md        the five defects
├── 04-tuning-methods.md       LoRA · QLoRA · SFT · DPO · distillation
└── 05-governance.md           what a review board actually asks
```

---

## Where this stops

- **The PII detectors are a first pass, not a compliance control.** They catch shapes that leak through copy-paste. Names, addresses and account references need a real NER or DLP pass — Purview, Macie, Cloud DLP. Knowing where the handoff is beats a tool that claims it doesn't need one.
- **Near-dedup is O(n²).** Correct, and honestly so. At scale you want MinHash LSH; swap the implementation, the interface doesn't change.
- **Nothing here calls a model.** You supply outputs to the eval harness. That's what keeps the scoring logic testable rather than being the one uninspected thing between you and a shipping decision.
- **This is not compliance advice.** It produces the artifacts your control framework needs, cheaply enough that nobody skips them.

---

## Contributing

Issues and PRs welcome — especially additional scorers, real-world curation defects this misses, and backends for the near-dedup path. Bring tests: every existing one is named for the mistake it prevents, and new controls should be mutation-checked the same way. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
