# Task #3 Review Notes: Final Evaluation Reproducibility

Reviewed branch: `final-evaluation` (at `d5c6428`). Nothing in the extraction model, pipeline, or existing reports was modified. The only additions are `tests/test_task3_reproducibility.py` and this file.

## What was checked

`tests/test_task3_reproducibility.py` reads the committed artifacts only. It does not load a model or rerun anything.

| # | Check | Result |
|---|-------|--------|
| 1 | `data/eval/test_set_neiss.csv` has 3,000 rows | Pass |
| 2 | Test `case_number` values are unique | Pass |
| 3 | Test `narrative` values are unique | Pass |
| 4 | Every test PSU (11 in total) is labeled `test` in `reports/split_by_psu.csv` | Pass |
| 5 | `reports/final_eval_metrics.csv` has exactly v1, v2, v3, keyword, majority | Pass |
| 6 | v3 metrics match the committed report (tolerance ±0.001) | Pass |
| 7 | `reports/final_eval_details.csv` has 3,000 rows, with the same case numbers as the test set | Pass |
| 8 | No tracked caches, model weights, files over 5 MB, `.env`, or credential patterns | Pass |

## Confirmed results

v3 on the 3,000-case test set: micro F1 0.739, macro F1 0.471, exact match 0.712.

| System | Micro F1 | Macro F1 | Exact match |
|--------|----------|----------|-------------|
| v1 | 0.591 | 0.434 | 0.506 |
| v2 | 0.623 | 0.435 | 0.575 |
| v3 | 0.739 | 0.471 | 0.712 |
| keyword | 0.851 | 0.623 | 0.851 |
| majority | 0.117 | 0.001 | 0.117 |

## Caveats

- **Product_1 is proxy ground truth.** NEISS `Product_1` is the product code chosen by the coder. It is not a label of the object named in the narrative. Scores measure agreement with that code, not object-level accuracy.
- **No new hand-labeled test set was used.** The test set is a sample from the NEISS data, restricted to the held-out test-fold PSUs. The 25 hand-labeled development cases are excluded from it.
- **The keyword baseline scores higher than v3.** It maps cue phrases straight to NEISS codes and does not extract objects. `reports/final_evaluation.md` discusses this. It should not be read as the keyword rule being the better extractor.

## Rerun not needed

These checks confirm the committed test set, split, metrics and details agree with each other and with the report. The extraction pipeline does not need to be rerun, and nothing was regenerated.

## Commands run

```
python -m pytest tests/test_task3_reproducibility.py   # 8 new checks
python -m pytest tests                                  # 75 passed (existing + new)
python -m py_compile tests/test_task3_reproducibility.py src/*.py verify_task2.py
```
