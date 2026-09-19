# Benchmark: normal RAG vs the RAPTOR tree

Questions: {"specific": 32}, generated and verified as described in `eval/benchmark.py`.

## Score by context budget

| Questions | Passages | `flat` | `tree_ext` | `tree_ext_fine` | `tree_llm` |
|---|---|---|---|---|---|
| specific | 4 | 0.688 | 0.656 | 0.656 | 0.656 |
| specific | 6 | 0.750 | 0.719 | 0.719 | 0.688 |
| specific | 8 | 0.781 | 0.781 | 0.781 | 0.781 |

## Difference from flat RAG, with 95% bootstrap intervals

| Questions | Passages | Method | Difference | 95% interval | Questions |
|---|---|---|---|---|---|
| specific | 4 | `tree_ext` | -0.031 | [-0.094, +0.000] | 32 |
| specific | 4 | `tree_ext_fine` | -0.031 | [-0.094, +0.000] | 32 |
| specific | 4 | `tree_llm` | -0.031 | [-0.094, +0.000] | 32 |
| specific | 6 | `tree_ext` | -0.031 | [-0.094, +0.000] | 32 |
| specific | 6 | `tree_ext_fine` | -0.031 | [-0.094, +0.000] | 32 |
| specific | 6 | `tree_llm` | -0.062 | [-0.156, +0.000] | 32 |
| specific | 8 | `tree_ext` | +0.000 | [+0.000, +0.000] | 32 |
| specific | 8 | `tree_ext_fine` | +0.000 | [+0.000, +0.000] | 32 |
| specific | 8 | `tree_llm` | +0.000 | [+0.000, +0.000] | 32 |

An interval that excludes zero is a difference unlikely to be noise.

## By document genre (6 passages)

| Genre | Questions | n | `flat` | `tree_ext` | `tree_ext_fine` | `tree_llm` |
|---|---|---|---|---|---|---|
| research paper | specific | 9 | 0.556 | 0.444 | 0.444 | 0.444 |
| survey | specific | 5 | 0.600 | 0.600 | 0.600 | 0.600 |
| policy framework | specific | 5 | 0.600 | 0.600 | 0.600 | 0.600 |
| study guide | specific | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| lab Q&A guide | specific | 4 | 1.000 | 1.000 | 1.000 | 0.750 |
| worked solutions | specific | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
