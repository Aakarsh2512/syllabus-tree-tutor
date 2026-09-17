# Evaluation results

Corpus: 5 documents, 192 leaf chunks, 202 tree nodes across 4 levels.

Questions: {"specific": 14, "broad": 8, "unanswerable": 4}. 
Answer provider: `ollama`. top_k = 6.

| Run | Specific hit@k | Specific MRR | Broad coverage | Refusal rate | Summary nodes / query | s per query |
|---|---|---|---|---|---|---|
| `flat` | 1.000 | 0.738 | 0.938 | 1.000 | 0.00 | 18.99 |
| `tree` | 1.000 | 0.726 | 0.938 | 1.000 | 0.96 | 13.62 |
| `tree_hybrid` | 0.929 | 0.786 | 0.875 | 1.000 | 0.92 | 22.95 |

Metric meanings are in `eval/run_eval.py`.
