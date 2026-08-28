# Contributing

Corrections especially welcome. This repository makes claims about how tuning
projects fail; if one of them is wrong in your experience, that is worth more
than a feature.

## What would help most

1. **Curation defects this misses.** The five in `curate.py` are the ones I
   have watched survive a training run. There are certainly more. Bring a
   failing test and the detector that fixes it.
2. **A scalable near-dedup backend.** The current implementation is O(n²) and
   says so. MinHash LSH behind the same interface would be a real improvement.
3. **More scorers.** `evaluate.py` covers the common answer shapes. Structured
   outputs, multi-field extraction and rubric scoring are all missing.
4. **Worked examples in other domains.** The bundled dataset is
   utility-operations flavoured. Healthcare, financial services and legal all
   have distinct curation failure modes.

## The bar for a new control

Anything presented as a control has to fail loudly when removed. Concretely:

- **A test named for the incident it prevents.** `test_decontaminate_works`
  tells a future reader nothing about whether it is safe to delete.
  `test_an_eval_example_in_the_training_set_is_removed` does.
- **Mutation-checked.** Break the control deliberately, confirm the suite goes
  red, and say so in the PR. A test that stays green when you break the thing
  it names is worse than no test, because it certifies the wrong thing.
- **No new runtime dependency in `src/`.** The toolkit is pure stdlib on
  purpose: that is what lets it run in milliseconds anywhere, which is what
  makes it get run.

Two of the thirteen mutations in the README initially survived, and both
turned out to be bugs in the *tests* rather than the code. That is the normal
outcome of doing this properly, and it is fine to report it that way.

## Running things

```bash
pip install -e ".[dev]"
pytest -q                                   # 130 tests, ~0.2s
ruff check src/ tests/ examples/
python examples/01-decide/decide.py
python examples/02-curate-a-dataset/curate_dataset.py
```

The training example needs a GPU and its own requirements; CI validates its
configs rather than running it.

## Style

Comments explain **why**, not what. If a line is a control, the comment says
what breaks when someone deletes it during a cleanup — that comment is often
the only thing standing between the control and a tidy-up.
