# Example 01 — Should we tune?

```bash
python examples/01-decide/decide.py
```

No GPU. No API key. Under a second.

Four scenarios. Three of them sound like fine-tuning problems in a planning
meeting; one is. The other three are a retrieval problem, an access problem,
and an economics problem wearing a quality problem's clothes.

Then the cost model, run at two volumes with everything else held constant:

| | 4,000,000 req/month | 25,000 req/month |
|---|---|---|
| Prompted, large model | $19,800 / mo | $124 / mo |
| Tuned, small model | $4,700 / mo | $3,508 / mo |
| Monthly saving | **$15,100** | **−$3,384** |
| One-time setup | $45,000 | $45,000 |
| **Breakeven** | **3.0 months** | **never** |

The low-volume column is the finding. Same per-token advantage, same setup
cost — and it never pays back, because the monthly cost of *owning* a tuned
model exceeds the entire saving. That line is missing from most business
cases, and it is usually the one that decides the answer.

**Read next:** [docs/01-when-to-tune.md](../../docs/01-when-to-tune.md)
