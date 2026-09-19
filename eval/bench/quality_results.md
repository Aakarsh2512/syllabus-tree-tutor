# QuALITY: normal RAG vs the RAPTOR tree

Human-written multiple-choice questions (4 options, so guessing scores 25%). 6 passages of context per question.

| Method | All | Difficult | Not difficult | Questions | Summaries used / question |
|---|---|---|---|---|---|
| `flat` | 60.0% | 43.3% | 76.7% | 60 | 0.00 |
| `tree_ext` | 65.0% | 50.0% | 80.0% | 60 | 1.83 |
| `tree_llm` | 71.7% | 53.3% | 90.0% | 60 | 2.98 |

## Paired comparison with flat RAG, same questions

| Method | Difference | 95% interval | Exact p | Both right | Only tree right | Only flat right | Both wrong |
|---|---|---|---|---|---|---|---|
| `tree_ext` | +5.0 pts | [-3.3, +13.3] | 0.453 | 34 | 5 | 2 | 19 |
| `tree_llm` | +11.7 pts | [+3.3, +21.7] | 0.039 ** | 35 | 8 | 1 | 16 |

The two middle columns are the real difference: questions one method answered and the other did not. The rest were answered the same way by both.
