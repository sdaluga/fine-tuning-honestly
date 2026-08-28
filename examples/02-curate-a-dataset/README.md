# Example 02 — Curate a dataset

```bash
python examples/02-curate-a-dataset/curate_dataset.py
```

No GPU. No API key. Under a second.

`data/train.jsonl` holds fourteen plausible-looking operational examples. It
is also seeded with every defect this toolkit exists to catch:

| Planted | Rows |
|---|---|
| Exact duplicate | 1 |
| Near-duplicates (case change, trailing punctuation) | 2 |
| Contaminates the eval set | 2 |
| Contains an email address and a phone number | 1 |
| Empty completion | 1 |

None of them announce themselves. Every one would train.

The run finds all of them, **blocks the dataset from release**, and then does
the part a person has to do: removes the offending rows by hand and writes
`data/train.curated.jsonl`. That step is deliberately outside `curate()` — a
scrubber you trust is a scrubber that will one day miss a row and not tell you.

It finishes by emitting the two artifacts a review board asks for:

- `output/dataset-manifest.json` — sources, checks run, and a content
  fingerprint that survives reshuffling but not editing
- `output/model-card.md` — which **lists its own gaps** rather than hiding them

**Read next:** [docs/03-data-curation.md](../../docs/03-data-curation.md)
