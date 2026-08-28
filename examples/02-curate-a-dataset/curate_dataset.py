#!/usr/bin/env python3
"""
Example 02 — Curate a real dataset, end to end.
===============================================

    python examples/02-curate-a-dataset/curate_dataset.py

No GPU. No API key. No network. Under a second.

`data/train.jsonl` is fourteen plausible-looking operational examples. It is
also seeded with every defect this toolkit exists to catch:

    * one exact duplicate
    * two near-duplicates (case change; trailing punctuation)
    * two rows that contaminate the eval set
    * one row containing an email address and a phone number
    * one row with an empty completion

None of these announce themselves. Every one of them would train.

The run ends by producing the two artifacts a review board asks for — a
dataset manifest with a content fingerprint, and a model card that lists its
own gaps rather than hiding them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from tuning_toolkit.curate import (  # noqa: E402
    Example,
    assert_no_leakage,
    curate,
    split,
)
from tuning_toolkit.governance import (  # noqa: E402
    ModelCard,
    build_manifest,
    reproducibility_check,
)

RULE = "=" * 74


def load(path: Path) -> list[Example]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        Example(prompt=r["prompt"], completion=r["completion"], source=r.get("source", "unknown"))
        for r in rows
    ]


def main() -> int:
    train = load(HERE / "data" / "train.jsonl")
    eval_set = load(HERE / "data" / "eval.jsonl")

    print(RULE)
    print("CURATING A TUNING DATASET")
    print(RULE)
    print(f"  loaded {len(train)} training rows and {len(eval_set)} eval rows")

    report = curate(train, eval_set)

    # ---- what was removed, and why ------------------------------------
    print(f"\n{RULE}")
    print("REMOVED")
    print("-" * 74)
    if not report.dropped:
        print("  nothing")
    for ex, reason in report.dropped:
        print(f"\n  [{ex.id}] {reason}")
        print(f"      source : {ex.source}")
        print(f"      prompt : {ex.prompt[:62]}")

    # ---- what needs a human -------------------------------------------
    print(f"\n{RULE}")
    print("FINDINGS")
    print("-" * 74)
    if not report.findings:
        print("  none")
    for f in sorted(report.findings, key=lambda x: (x.severity != "block", x.kind)):
        mark = "BLOCK" if f.severity == "block" else " warn"
        print(f"  [{mark}] {f.kind:<26} {f.detail}")

    # ---- the gate ------------------------------------------------------
    print(f"\n{RULE}")
    print("RELEASE GATE")
    print("-" * 74)
    s = report.summary()
    print(f"  kept {s['kept']}  ·  dropped {s['dropped']}  ·  "
          f"findings {s['findings']}  ·  blocking {s['blocking']}")
    print()
    if report.is_releasable:
        print("  PASS — this dataset may proceed to training.")
    else:
        print("  BLOCKED — this dataset does not get trained on.")
        print()
        print("  Not 'trained on with a note in the ticket'. The PII row is the")
        print("  reason: once it is in the weights it is extractable, permanent,")
        print("  and beyond the reach of any deletion request. This is the last")
        print("  cheap moment to remove it.")

    # ---- the human decision ---------------------------------------------
    # This toolkit flags; it does not silently fix. What follows is the step a
    # PERSON takes after reading the findings above: remove the offending rows
    # and write a curated file, with the removal recorded rather than implied.
    #
    # Doing it here, in the example, rather than inside `curate()` is the whole
    # design. A scrubber you trust is a scrubber that will one day miss a row
    # and never tell you.
    blocked_ids = {f.example_id for f in report.blocking}
    curated = [e for e in report.kept if e.id not in blocked_ids]

    curated_path = HERE / "data" / "train.curated.jsonl"
    curated_path.write_text(
        "\n".join(
            json.dumps({"prompt": e.prompt, "completion": e.completion, "source": e.source})
            for e in curated
        )
        + "\n"
    )
    print()
    print(f"  After review, {len(blocked_ids)} row(s) removed by hand ->")
    print(f"  data/{curated_path.name}  ({len(curated)} examples)")

    # ---- split integrity ------------------------------------------------
    train_split, held = split(report.kept, eval_fraction=0.2)
    leaks = assert_no_leakage(train_split, held)
    print(f"\n{RULE}")
    print("SPLIT")
    print("-" * 74)
    print(f"  train {len(train_split)}  ·  held-out {len(held)}")
    print(f"  leakage check: {'CLEAN' if not leaks else f'{len(leaks)} LEAKS'}")
    print("  (content-hash split — identical text always lands on the same side,")
    print("   so a duplicate that survived dedup cannot straddle the boundary)")

    # ---- artifacts -------------------------------------------------------
    manifest = build_manifest(
        name="ops-triage-sft",
        version="0.1.0",
        report=report,
        checks_run=["exact-dedup", "near-dedup", "decontamination", "pii-scan", "format"],
    )

    card = ModelCard(
        model_name="ops-triage-small",
        base_model="<base-model-id>",
        version="0.1.0",
        owner="Platform AI",
        intended_use=(
            "Classify inbound operational messages by urgency and draft a "
            "suggested reply for human review. Never sends."
        ),
        method="lora",
        dataset=manifest,
        # Deliberately left incomplete — see below.
        hyperparameters={"rank": 16, "alpha": 32, "lr": 1e-4, "epochs": 3},
    )

    out = HERE / "output"
    out.mkdir(exist_ok=True)
    (out / "dataset-manifest.json").write_text(manifest.to_json())
    (out / "model-card.md").write_text(card.to_markdown())

    print(f"\n{RULE}")
    print("ARTIFACTS")
    print("-" * 74)
    print(f"  {out.name}/dataset-manifest.json")
    print(f"  {out.name}/model-card.md")
    print()
    print(f"  fingerprint  {manifest.fingerprint[:32]}…")
    print("  Sorted content hashes, so reshuffling the data does not change its")
    print("  identity — but editing one row does. That is what makes it evidence.")

    print()
    print("  The card reports its own gaps rather than hiding them:")
    for gap in card.gaps():
        print(f"    - {gap}")
    print()
    print("  That list is the feature. A card with no limitations section tells")
    print("  a reviewer nobody looked, and they are right. Run gaps() in CI and")
    print("  the card cannot rot — the build fails before a meeting is booked.")

    # ---- reproducibility -------------------------------------------------
    tampered = report.kept[:-1] + [
        Example("Summarise the vibration trend on Unit 1 bearing 4.", "QUIETLY EDITED", "x")
    ]
    print(f"\n{RULE}")
    print("REPRODUCIBILITY")
    print("-" * 74)
    print(f"  original dataset : {'verifies' if not reproducibility_check(card, report.kept) else 'MISMATCH'}")
    for p in reproducibility_check(card, tampered):
        print(f"  tampered dataset : {p}")

    print(f"\n{RULE}")
    print("Read next: docs/03-data-curation.md")
    print(RULE)
    return 0 if report.is_releasable else 0  # reporting run: never fails the shell


if __name__ == "__main__":
    raise SystemExit(main())
