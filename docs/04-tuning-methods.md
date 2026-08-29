# 04 · Tuning methods

You've climbed the ladder, the diagnosis is behavioural or economic, the data is curated, and the eval is committed with a scored baseline. Now — and only now — which method?

If any of those four is not true, the answer is not on this page. Go back to [01](01-when-to-tune.md), [02](02-evaluation-first.md) or [03](03-data-curation.md).

---

## The short version

| Method | Use when | Trains | Typical hardware |
|---|---|---|---|
| **LoRA** | Style, format, narrow behaviour | ~0.1–1% of params | 1× 24–48GB |
| **QLoRA** | Same, on a bigger base or smaller card | Same, 4-bit frozen base | 1× 24GB |
| **Full SFT** | Broad behaviour change, plenty of data | 100% | Multi-GPU |
| **Distillation** | Quality is fine, cost is not | Small model, teacher outputs | 1× 24–48GB |
| **DPO** | Taste you can rank but not state | Adapter or full | 1× 48GB+ |
| **Continued pretraining** | Genuinely novel domain vocabulary | 100%, huge corpus | Cluster |

**Start with LoRA.** It is reversible — the adapter is a separate artifact you can detach — cheap enough to iterate on, and sufficient for the large majority of enterprise adaptation. Reach past it when a held-out curve says to.

---

## LoRA, and the knob everyone over-tunes

LoRA freezes the base model and trains small low-rank matrices alongside the attention and MLP projections. The adapter is typically tens of megabytes against tens of gigabytes of base.

That separability is worth more than the compute saving: you can ship an adapter, roll it back, keep several for different tasks against one base, and — for a governance review — point at exactly what changed.

### Rank

The most consequential knob and the most over-tuned.

| Rank | Fits |
|---|---|
| 4–8 | Style and tone; format compliance |
| **8–16** | **The default. Most adaptation lives here** |
| 32–64 | Genuinely new behaviour, thousands of examples |
| 128+ | You probably want full SFT, and should say so |

Raising rank because "more is better" buys overfitting on a small dataset and a slower merge. **Move it when a held-out loss curve tells you to.**

### Alpha

The effective scale is `alpha / rank`. Convention is `alpha = 2 × rank`. Move one, not both, or you cannot attribute the result. Alpha *below* rank scales your adapter down and is almost always a typo — the config validator flags it.

### Target modules

Attention only (`q,k,v,o`) is cheaper and frequently enough for restyling. Add the MLP projections (`gate,up,down`) when teaching genuinely new behaviour rather than reshaping existing behaviour.

### Learning rate and epochs

LoRA tolerates a learning rate roughly **10× a full fine-tune's**, because only the adapters move — `1e-4` to `2e-4` is the usual band. **Two to three epochs.** Past that, on a curated set, you are memorising, and your eval is too small to tell you. Watch held-out loss, not training loss.

---

## QLoRA

LoRA with the base model quantised to 4-bit. Fits a 7–8B tune on a single 24GB card at a quality cost that is usually invisible for adaptation work.

The trade is speed: dequantisation on every forward pass costs perhaps 20–30% throughput. Take it when memory is the binding constraint, skip it when it isn't.

---

## Distillation

The most defensible reason to change weights and the least discussed.

You are not making a model smarter. You are teaching a small model to imitate a large one **on one narrow task**, which is a far easier target than general capability. Generate outputs from the large model, curate them exactly as you would human labels — they carry the teacher's mistakes too — and fine-tune the small one.

This is where the economics in [01](01-when-to-tune.md) actually land, and it is the most common genuine win in enterprise settings.

---

## Preference tuning (DPO)

For targets you can **rank but not state**. If you can articulate the rule, put it in the prompt — that's reversible and free.

Needs ranked pairs, which are more expensive than they look: two generations and a human judgement each. Budget ~1,000 pairs minimum, and remember that the judgements encode whoever made them.

---

## Continued pretraining

Rare, expensive, and usually proposed for the wrong reason. "Our company has a specific tone" is not a novel domain. Some proprietary industrial and scientific corpora genuinely are.

Prove it before you propose it: show the base model failing on domain terms it should recognise.

---

## What the configs look like

```bash
# validate a config and the dataset — no GPU needed
python examples/04-lora-finetune/train_lora.py \
    --config examples/04-lora-finetune/configs/lora-7b.yaml \
    --validate-only
```

Four configs ship (the fourth, `lora-small-fast.yaml`, is for iterating on the data rather than the adapter):

- **`lora-7b.yaml`** — the defensible default. Points at the *raw* dataset, so it demonstrates the refusal path.
- **`lora-7b-curated.yaml`** — the same run against the reviewed dataset. Validates clean.
- **`BROKEN-example.yaml`** — deliberately misconfigured, so the validator has something to catch in CI. Every mistake in it has cost somebody a GPU-hour.

The training script **refuses to run** on an uncurated dataset. That is a hard stop, not a flag you can pass. Every hour of GPU spent on contaminated data produces a model whose score you cannot interpret.

---

## The ratio is the argument

The training script is 344 lines, and most of that is comments explaining why each hyperparameter has the value it does. The toolkit around it — decision, curation, evaluation, governance — is roughly 1,500 lines and 130 tests.

That ratio is deliberate and it is the point of this repository. The training loop is close to a solved problem: `peft` and `trl` do the work, and a wrong hyperparameter usually announces itself within the hour. The parts that are **not** solved — is this the right rung, is the data clean, did anything regress, can you prove what went in — fail silently, months later, in front of someone who is not on your team.

---

**Next:** [05 · Governance](05-governance.md)
