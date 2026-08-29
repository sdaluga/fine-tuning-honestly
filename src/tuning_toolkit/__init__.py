"""
tuning_toolkit — the parts of a fine-tuning project that must be correct
whether or not the training run works.

Four modules, all pure, all runnable with no GPU, no API key and no network:

    decision    should you tune at all, and what does it cost
    curate      dedup, decontamination, PII detection, split integrity
    evaluate    the eval harness and the regression gate
    distill     teacher-label quality, and the agreement trap
    governance  dataset manifests, model cards, reproducibility checks

The training script itself lives in examples/04-lora-finetune and needs a GPU.
It is deliberately the smallest, least interesting file in the repository.
That ratio is the argument this project is making.
"""

from .decision import (  # noqa: F401
    CostInputs,
    Recommendation,
    Rung,
    Scenario,
    breakeven_months,
    cost_summary,
    recommend,
)
from .curate import (  # noqa: F401
    CurationReport,
    Example,
    Finding,
    assert_no_leakage,
    curate,
    decontaminate,
    detect_pii,
    exact_dedup,
    near_dedup,
    scan_pii,
    split,
    validate_format,
)
from .evaluate import (  # noqa: F401
    Case,
    EvalRun,
    GateResult,
    contains,
    exact_match,
    format_report,
    json_field,
    normalized_match,
    numeric_close,
    regex_match,
    regression_gate,
    run_eval,
)
from .distill import (  # noqa: F401
    DEFAULT_MIN_AGREEMENT,
    Candidate,
    DistillReport,
    accuracy_against_gold,
    agreement_rate,
    expected_accuracy,
    filter_by_consensus,
    label_accuracy,
    majority_vote,
    to_examples,
)
from .governance import (  # noqa: F401
    DatasetManifest,
    ModelCard,
    build_manifest,
    fingerprint_examples,
    reproducibility_check,
)

__version__ = "0.1.0"
