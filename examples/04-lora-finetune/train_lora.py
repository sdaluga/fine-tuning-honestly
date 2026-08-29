#!/usr/bin/env python3
"""
Example 03 — The actual LoRA fine-tune.
=======================================

    pip install -r examples/04-lora-finetune/requirements.txt
    python examples/04-lora-finetune/train_lora.py --config configs/lora-7b.yaml

NEEDS A GPU. This is the one thing in the repository that does. CI validates
its syntax and its configs; CI does not run it, because a training job that
runs on every push is a bill, not a test.

WHY THIS IS THE SHORTEST INTERESTING FILE HERE
----------------------------------------------
That ratio is the argument. This file is ~340 lines, most of them comments
explaining why a hyperparameter has the value it does, against roughly 1,500
lines and 130 tests of decision, curation, evaluation and governance — and
this file is the part everyone pictures when they say "we fine-tuned a model."

The training loop is close to a solved problem. `peft` and `trl` do the work,
the hyperparameters below are defensible defaults, and if you get one wrong
you will usually find out in an hour. The parts that are NOT solved — is this
the right rung, is the data clean, did anything regress, can you prove what
went in — are the parts that fail silently, months later, in front of someone
who is not on your team.

WHAT THIS SCRIPT REFUSES TO DO
------------------------------
It will not train on a dataset that has not been curated, and it will not
train without a baseline eval. Both are hard stops, not warnings. Every hour
of GPU spent on a contaminated dataset produces a model whose score you
cannot interpret, and the most expensive version of that mistake is the one
you only notice after it ships.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tuning_toolkit.curate import Example, curate  # noqa: E402
from tuning_toolkit.governance import build_manifest  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    """
    Every field here is a decision, and the comment is the reasoning.

    The defaults are a starting point for an instruction-following tune on a
    7-8B base. They are not universal, and anyone who tells you their
    hyperparameters are universal is selling something.
    """

    base_model: str
    train_file: str
    eval_file: str
    output_dir: str

    # ---- LoRA ------------------------------------------------------------
    # Rank. The single most consequential knob, and the most over-tuned.
    # 8-16 covers most style and format adaptation. Going to 64 because
    # "more is better" mostly buys overfitting on a small dataset and a
    # longer merge. Raise it when a held-out curve says to, not before.
    rank: int = 16

    # Convention is alpha = 2 * rank; the effective scale is alpha/rank.
    # Move one, not both, or you cannot attribute the change.
    alpha: int = 32

    dropout: float = 0.05

    # Which projections get adapters. Attention-only (q,k,v,o) is cheaper and
    # often enough. Adding the MLP projections helps when you are teaching
    # genuinely new behaviour rather than restyling existing behaviour.
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # ---- Optimisation -----------------------------------------------------
    # LoRA tolerates a learning rate roughly 10x a full fine-tune's, because
    # only the adapters move. 1e-4 to 2e-4 is the usual band.
    learning_rate: float = 1e-4

    # Two to three epochs. Beyond that, on a small curated set, you are
    # memorising — and the eval will not tell you, because the eval is small
    # too. Watch the held-out loss, not the training loss.
    epochs: int = 3

    batch_size: int = 4
    grad_accum: int = 4  # effective batch 16
    max_seq_len: int = 2048
    warmup_ratio: float = 0.03

    # 4-bit base weights (QLoRA). Fits a 7B tune on a single 24GB card at a
    # quality cost that is usually invisible for adaptation work.
    load_in_4bit: bool = True

    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LoRAConfig":
        import yaml  # imported here so config validation needs no torch

        data = yaml.safe_load(Path(path).read_text())
        if "target_modules" in data:
            data["target_modules"] = tuple(data["target_modules"])
        return cls(**data)

    def validate(self) -> list[str]:
        """
        Catch the misconfigurations that waste a GPU-hour before you spend it.

        Cheap, and it runs in CI — which is why every one of these is a
        returned message rather than an exception raised three minutes into a
        job you are no longer watching.
        """
        problems: list[str] = []
        if self.rank < 1:
            problems.append(f"rank must be >= 1, got {self.rank}")
        if self.rank > 128:
            problems.append(
                f"rank {self.rank} is very high for adaptation; expect overfitting "
                f"on a small dataset — justify it with a held-out curve"
            )
        if not 0.0 <= self.dropout < 1.0:
            problems.append(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0 < self.learning_rate < 1e-2:
            problems.append(f"learning_rate {self.learning_rate} is outside any sane band")
        if self.epochs > 5:
            problems.append(
                f"{self.epochs} epochs on a curated set is memorisation, not learning"
            )
        if self.alpha < self.rank:
            problems.append(
                f"alpha ({self.alpha}) below rank ({self.rank}) scales the adapter "
                f"down; almost always a typo"
            )
        if not self.target_modules:
            problems.append("no target_modules — there is nothing to adapt")
        return problems


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[Example]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        Example(r["prompt"], r["completion"], r.get("source", "unknown")) for r in rows
    ]


def gate_dataset(train_path: Path, eval_path: Path) -> tuple[list[Example], dict]:
    """
    Refuse to train on a dataset that has not passed curation.

    A hard stop, deliberately. The alternative — a warning nobody reads at
    3am — produces a model whose eval number is uninterpretable, and the
    expensive version of that mistake is the one discovered after release.
    """
    train = load_jsonl(train_path)
    eval_set = load_jsonl(eval_path)

    report = curate(train, eval_set)

    if not report.is_releasable:
        print("REFUSING TO TRAIN — the dataset has blocking findings:\n")
        for f in report.blocking:
            print(f"  [{f.kind}] {f.detail}")
        print("\nFix the data. Do not pass a flag to skip this.")
        raise SystemExit(2)

    manifest = build_manifest(
        name=train_path.stem,
        version="from-run",
        report=report,
        checks_run=["exact-dedup", "near-dedup", "decontamination", "pii-scan", "format"],
    )
    print(f"dataset OK — {len(report.kept)} examples, "
          f"{len(report.dropped)} removed, fingerprint {manifest.fingerprint[:16]}…")
    return report.kept, json.loads(manifest.to_json())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(cfg: LoRAConfig, examples: list[Example], manifest: dict) -> None:
    """
    The training run. Imports live inside the function on purpose, so that
    `--validate-only` and the CI config check never need torch installed.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    torch.manual_seed(cfg.seed)

    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        if cfg.load_in_4bit
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        # Without this the collator pads with EOS and the model learns to
        # emit EOS early — a short-output regression that is maddening to
        # diagnose after the fact.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=list(cfg.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    def to_text(ex: Example) -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": ex.prompt},
                {"role": "assistant", "content": ex.completion},
            ],
            tokenize=False,
        )

    dataset = Dataset.from_dict({"text": [to_text(e) for e in examples]})

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Written BEFORE training, so an interrupted run still leaves a record of
    # exactly what it was going to consume.
    (out / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2))

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            seed=cfg.seed,
            report_to="none",
        ),
        max_seq_length=cfg.max_seq_len,
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)

    print(f"\nadapter written to {cfg.output_dir}")
    print("NEXT, AND NOT OPTIONAL: run the eval and the regression gate.")
    print("An adapter that has not been gated is not a result, it is a file.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LoRA fine-tune with curation and config gates.")
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="check the config and the dataset, then stop. No GPU needed.",
    )
    args = ap.parse_args(argv)

    cfg = LoRAConfig.from_yaml(args.config)

    if problems := cfg.validate():
        print("config problems:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"config OK — rank {cfg.rank}, alpha {cfg.alpha}, lr {cfg.learning_rate}, "
          f"{cfg.epochs} epochs")

    root = Path(args.config).resolve().parent
    examples, manifest = gate_dataset(root / cfg.train_file, root / cfg.eval_file)

    if args.validate_only:
        print("\n--validate-only: stopping before the GPU work.")
        return 0

    train(cfg, examples, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
