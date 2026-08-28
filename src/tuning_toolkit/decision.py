"""
Should you fine-tune? Usually not — and this module says so with numbers.
=======================================================================

THE PROBLEM THIS SOLVES
-----------------------
Fine-tuning is the most requested and least justified step in applied AI. It
gets proposed because it sounds like the serious answer, and because "we
trained our own model" survives a steering-committee slide better than "we
wrote a better prompt."

The honest sequence is a ladder. Each rung costs roughly an order of magnitude
more than the one below it, in engineering time far more than in GPU spend,
and each one is only worth climbing when the rung below has demonstrably run
out of room:

    0  PROMPT          instructions, format, examples in context
    1  FEW_SHOT        curated exemplars, chosen not guessed
    2  RETRIEVAL       give it the facts it lacks (RAG)
    3  TOOLS           let it call the system of record
    4  DISTILL         big model's outputs -> small model, for cost/latency
    5  SFT             supervised fine-tune, for behaviour prompts can't reach
    6  PREFERENCE      DPO/RLHF, for taste you can rank but not describe
    7  CONTINUED_PT    continued pretraining, for a genuinely new domain

The two most common real wins are rung 2 (the model didn't know the fact) and
rung 4 (the model knew fine, you just couldn't afford it at volume). Rung 5
is legitimate but narrower than its reputation. Rungs 6 and 7 are rare enough
in enterprise settings that proposing one should require defending it.

WHAT THIS MODULE IS NOT
-----------------------
Not a scoring rubric that launders a decision you already made. Every
recommendation carries the specific reason, and `blockers` lists the things
that must be true before tuning is even a coherent option. If the blockers
are non-empty, the answer is "not yet" regardless of how good the case is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Rung(IntEnum):
    """The ladder. Ordered, so `>` means "more expensive and more invasive"."""

    PROMPT = 0
    FEW_SHOT = 1
    RETRIEVAL = 2
    TOOLS = 3
    DISTILL = 4
    SFT = 5
    PREFERENCE = 6
    CONTINUED_PT = 7


#: Everything at or above this rung changes model weights, which is what turns
#: an engineering decision into a governance one: a tuned model needs a data
#: lineage record, an approval, and a place in the model inventory.
FIRST_WEIGHT_CHANGING_RUNG = Rung.DISTILL


@dataclass(frozen=True)
class Scenario:
    """
    What you actually know about the problem.

    Deliberately small. If a field here is a guess, the recommendation is a
    guess, and you should go measure instead of running this function.
    """

    #: Does the base model fail because it lacks FACTS it could be handed?
    #: If yes, tuning is close to the worst available answer: you would be
    #: baking a snapshot of a moving corpus into weights.
    failure_is_missing_knowledge: bool = False

    #: Does it fail because it lacks ACCESS — it needs to read or write a
    #: system of record?
    failure_is_missing_access: bool = False

    #: Does it fail on FORMAT or STYLE that prompting has genuinely not fixed?
    failure_is_behaviour: bool = False

    #: Can you rank two outputs as better/worse but not write down the rule?
    #: This, and essentially only this, is what preference tuning is for.
    failure_is_unstateable_taste: bool = False

    #: Is the model's quality acceptable and the problem purely cost/latency?
    quality_is_fine_cost_is_not: bool = False

    #: Have you actually iterated on the prompt? Be honest. "We tried a few
    #: things" is not the same as a systematic pass with an eval set.
    prompt_iterations: int = 0

    #: Number of high-quality, human-reviewed labelled examples ON HAND.
    #: Not "we could probably generate some."
    labelled_examples: int = 0

    #: Do you have an eval set that would DETECT the improvement you want?
    has_eval_set: bool = False

    #: Is the domain vocabulary genuinely absent from public pretraining data?
    #: True for some proprietary industrial and scientific corpora. Almost
    #: never true for "our company's tone of voice."
    domain_is_novel: bool = False

    #: Monthly request volume, for the cost model.
    monthly_requests: int = 0


@dataclass(frozen=True)
class Recommendation:
    rung: Rung
    reason: str
    #: Things that must be true before a weight-changing rung is even coherent.
    blockers: list[str] = field(default_factory=list)
    #: Cheaper rungs that should be exhausted first, in order.
    try_first: list[Rung] = field(default_factory=list)

    @property
    def changes_weights(self) -> bool:
        return self.rung >= FIRST_WEIGHT_CHANGING_RUNG

    @property
    def is_actionable(self) -> bool:
        """A recommendation with blockers is a 'not yet', not a 'go'."""
        return not self.blockers


#: Below this, a supervised fine-tune is fitting noise. The number is a floor,
#: not a target, and it assumes the examples are human-reviewed. A thousand
#: scraped rows is worth less than a hundred curated ones.
MIN_SFT_EXAMPLES = 500

#: Preference tuning needs pairs, and pairs are more expensive to produce than
#: they look — each one needs two generations and a human judgement.
MIN_PREFERENCE_PAIRS = 1_000

#: Fewer than this and "we tried prompting" is not a finding.
MIN_PROMPT_ITERATIONS = 3


def recommend(s: Scenario) -> Recommendation:
    """
    Return the lowest rung that could plausibly solve the stated problem.

    Order matters here and it is not arbitrary: the checks run cheapest-first,
    so a scenario that is really a retrieval problem can never fall through
    into a tuning recommendation just because it also mentions volume.
    """
    blockers = _blockers(s)

    # --- Rung 2: it doesn't know the fact -------------------------------
    # Checked first because it is both the most common real cause and the one
    # most often misdiagnosed as a tuning problem. Weights are a bad place to
    # put facts: they go stale, they cannot be cited, and correcting one
    # means retraining.
    if s.failure_is_missing_knowledge:
        return Recommendation(
            rung=Rung.RETRIEVAL,
            reason=(
                "The failure is missing knowledge, not missing capability. "
                "Fine-tuning would freeze a moving corpus into weights, with "
                "no citation path and no way to correct one fact without a "
                "retrain."
            ),
            try_first=[Rung.PROMPT, Rung.FEW_SHOT],
        )

    # --- Rung 3: it can't reach the system of record ---------------------
    if s.failure_is_missing_access:
        return Recommendation(
            rung=Rung.TOOLS,
            reason=(
                "The failure is missing access. No amount of training teaches "
                "a model what today's balance is; give it the call."
            ),
            try_first=[Rung.RETRIEVAL],
        )

    # --- Rung 4: quality is fine, economics are not ----------------------
    # The most under-used legitimate reason to change weights. You are not
    # trying to make the model smarter; you are trying to make a small model
    # imitate a big one on one narrow task, which is a far easier target.
    if s.quality_is_fine_cost_is_not:
        return Recommendation(
            rung=Rung.DISTILL,
            reason=(
                "Quality is acceptable and the constraint is cost or latency. "
                "Distillation targets exactly that: teach a small model to "
                "imitate the large one on this task only. This is the most "
                "defensible reason to change weights."
            ),
            blockers=blockers,
            try_first=[Rung.PROMPT, Rung.RETRIEVAL],
        )

    # --- Rung 7: genuinely novel domain ----------------------------------
    # Before rung 6, because a novel domain is a different (and rarer) claim
    # than unstateable taste, and continued pretraining subsumes it.
    if s.domain_is_novel and s.labelled_examples >= MIN_SFT_EXAMPLES:
        return Recommendation(
            rung=Rung.CONTINUED_PT,
            reason=(
                "The domain vocabulary is genuinely absent from pretraining. "
                "This is rare and expensive — confirm it by showing the base "
                "model failing on domain terms it should recognise, not by "
                "asserting the domain is special."
            ),
            blockers=blockers,
            try_first=[Rung.RETRIEVAL, Rung.SFT],
        )

    # --- Rung 6: you can rank it but not describe it ---------------------
    if s.failure_is_unstateable_taste:
        pref_blockers = list(blockers)
        if s.labelled_examples < MIN_PREFERENCE_PAIRS:
            pref_blockers.append(
                f"preference tuning needs ~{MIN_PREFERENCE_PAIRS} ranked pairs; "
                f"you have {s.labelled_examples}"
            )
        return Recommendation(
            rung=Rung.PREFERENCE,
            reason=(
                "The target is a preference you can rank but not state as a "
                "rule. That is what DPO is for. If you CAN state the rule, "
                "state it in the prompt instead — it is reversible."
            ),
            blockers=pref_blockers,
            try_first=[Rung.PROMPT, Rung.SFT],
        )

    # --- Rung 5: behaviour prompting couldn't reach ----------------------
    if s.failure_is_behaviour:
        return Recommendation(
            rung=Rung.SFT,
            reason=(
                "The failure is behavioural and survives prompt iteration. "
                "A supervised fine-tune is the right tool, and the work is "
                "almost entirely in the data, not the hyperparameters."
            ),
            blockers=blockers,
            try_first=[Rung.PROMPT, Rung.FEW_SHOT],
        )

    # --- Rungs 0-1: the default, and usually the answer -------------------
    return Recommendation(
        rung=Rung.FEW_SHOT if s.prompt_iterations >= MIN_PROMPT_ITERATIONS else Rung.PROMPT,
        reason=(
            "No stated failure mode requires changing weights. Curated "
            "in-context examples are reversible, auditable, and shippable "
            "this afternoon."
        ),
    )


def _blockers(s: Scenario) -> list[str]:
    """
    Preconditions for any weight-changing rung.

    These are not soft advice. Tuning without an eval set is not a risky
    experiment — it is an unmeasurable one, which is worse, because you will
    ship it anyway and attribute whatever happens next to it.
    """
    out: list[str] = []

    if not s.has_eval_set:
        out.append(
            "no eval set — you cannot detect the improvement you are paying for, "
            "and you will not notice the regression you bought with it"
        )

    if s.prompt_iterations < MIN_PROMPT_ITERATIONS:
        out.append(
            f"only {s.prompt_iterations} prompt iterations — "
            f"exhaust the free rung before paying for the expensive one"
        )

    if s.labelled_examples < MIN_SFT_EXAMPLES:
        out.append(
            f"{s.labelled_examples} labelled examples is below the ~{MIN_SFT_EXAMPLES} "
            f"floor; below this you are fitting noise and calling it a model"
        )

    return out


# ---------------------------------------------------------------------------
# The cost model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostInputs:
    """
    Per-unit costs. Supply your own — published prices move, and the point of
    this model is the SHAPE of the curves, not any particular vendor's rate.
    """

    monthly_requests: int
    #: Average tokens in / out per request.
    input_tokens: int
    output_tokens: int
    #: Cost per million tokens on the large model you would otherwise call.
    large_in_per_m: float
    large_out_per_m: float
    #: Same for the small model you would tune.
    small_in_per_m: float
    small_out_per_m: float
    #: One-time cost of producing the tuning set and running the job. This is
    #: overwhelmingly HUMAN time, not GPU time — the compute is usually the
    #: rounding error, which is the opposite of what people expect.
    tuning_setup_cost: float
    #: Monthly cost of owning a tuned model: eval reruns, drift checks,
    #: re-tunes when the base model version moves. The line people forget.
    tuning_monthly_overhead: float


def _monthly(reqs: int, tin: int, tout: int, in_rate: float, out_rate: float) -> float:
    return reqs * (tin * in_rate + tout * out_rate) / 1_000_000


def monthly_cost_prompted(c: CostInputs) -> float:
    """Keep calling the large model."""
    return _monthly(
        c.monthly_requests, c.input_tokens, c.output_tokens,
        c.large_in_per_m, c.large_out_per_m,
    )


def monthly_cost_tuned(c: CostInputs) -> float:
    """Call the tuned small model — plus the overhead of owning it."""
    return _monthly(
        c.monthly_requests, c.input_tokens, c.output_tokens,
        c.small_in_per_m, c.small_out_per_m,
    ) + c.tuning_monthly_overhead


def breakeven_months(c: CostInputs) -> float | None:
    """
    Months until tuning has paid back its setup cost.

    Returns None when it never does — which is the answer more often than
    teams expect, because `tuning_monthly_overhead` can eat the entire
    per-token saving at low volume. A None here is the single most useful
    output of this whole module.
    """
    saving = monthly_cost_prompted(c) - monthly_cost_tuned(c)
    if saving <= 0:
        return None
    return c.tuning_setup_cost / saving


def cost_summary(c: CostInputs) -> dict[str, float | None]:
    """Everything the slide needs, in one call."""
    prompted = monthly_cost_prompted(c)
    tuned = monthly_cost_tuned(c)
    return {
        "monthly_prompted": prompted,
        "monthly_tuned": tuned,
        "monthly_saving": prompted - tuned,
        "setup_cost": c.tuning_setup_cost,
        "breakeven_months": breakeven_months(c),
    }
