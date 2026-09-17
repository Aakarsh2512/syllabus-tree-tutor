# Evaluation results

Corpus: 5 documents, 192 leaf chunks, 202 tree nodes across 4 levels.

Questions: {"specific": 14, "broad": 8, "unanswerable": 4}. 
Answer provider: `offline`. top_k = 6.

| Run | Specific hit@k | Specific MRR | Broad coverage | Refusal rate | Summary nodes / query | s per query |
|---|---|---|---|---|---|---|
| `flat` | 1.000 | 0.738 | 0.938 | 0.750 | 0.00 | 1.04 |
| `tree` | 1.000 | 0.726 | 0.938 | 0.500 | 0.77 | 0.02 |
| `tree_hybrid` | 0.929 | 0.786 | 0.812 | 0.250 | 0.50 | 0.05 |

Metric meanings are in `eval/run_eval.py`.
