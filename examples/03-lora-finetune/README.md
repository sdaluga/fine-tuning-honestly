# Example 03 — The actual LoRA fine-tune

**Needs a GPU.** The only thing here that does.

```bash
pip install -r examples/03-lora-finetune/requirements.txt

# validate a config and the dataset — no GPU needed
python examples/03-lora-finetune/train_lora.py \
    --config examples/03-lora-finetune/configs/lora-7b-curated.yaml --validate-only

# the real thing
python examples/03-lora-finetune/train_lora.py \
    --config examples/03-lora-finetune/configs/lora-7b-curated.yaml
```

## Two hard stops

The script **refuses to train** on a dataset with blocking curation findings,
and refuses to run a config that fails validation. Neither is a flag you can
pass. Every GPU-hour spent on contaminated data produces a model whose score
you cannot interpret, and the expensive version of that mistake is the one
found after release.

```
config OK — rank 16, alpha 32, lr 0.0001, 3 epochs
REFUSING TO TRAIN — the dataset has blocking findings:
  [format:empty_completion] completion is empty — teaches the model that silence is valid
  [pii:email] email pattern found in example from 'contact-sheet'
  [pii:us_phone] us_phone pattern found in example from 'contact-sheet'

Fix the data. Do not pass a flag to skip this.
```

## The configs

| Config | Asserts |
|---|---|
| `lora-7b.yaml` | The defensible default, pointed at the **raw** dataset — demonstrates the refusal |
| `lora-7b-curated.yaml` | The same run against the reviewed dataset — validates clean |
| `BROKEN-example.yaml` | Deliberately misconfigured, so the validator has something to catch in CI |

CI runs all three and asserts a different exit code for each.

## Why this file is short

344 lines, most of them comments explaining why each hyperparameter has the
value it does — against roughly 1,500 lines and 130 tests of decision,
curation, evaluation and governance.

That ratio is the point of the repository. The training loop is close to
solved; `peft` and `trl` do the work and a wrong hyperparameter usually
announces itself within the hour. The parts that are *not* solved fail
silently, months later, in front of someone who is not on your team.

**Read next:** [docs/04-tuning-methods.md](../../docs/04-tuning-methods.md)
