# Final evaluation on a held-out test set

## Summary

- The final model (v3) reaches **micro F1 0.74**, **macro F1 0.47** and **exact-match accuracy 0.71** on 3,000 held-out narratives.
- It beats the v1 baseline on every metric. Micro F1 rises by **0.15** (95% interval 0.13 to 0.16) and exact match by **0.21**.
- About half of v3's counted errors are not model mistakes: they are sibling NEISS codes, products named alongside others, or objects that cannot be read from the text.

![Micro and macro F1 by system](figures/final_eval_comparison.png)

## Test set

- **Source:** 3,000 narratives sampled at random from the 11 test-fold hospitals in `reports/split_by_psu.csv` (`data/eval/test_set_neiss.csv`). No earlier task tuned on these hospitals.
- **Excluded:** the 25 hand-labeled development cases, the 300-row sample used to write the keyword rule, and any narrative whose text also appears in a train or validation hospital.
- **Ground truth:** the NEISS product code (`Product_1`).
- **Scoring a span:** v3 outputs text ("STEPS"), so each span is mapped to a code by a lexicon built from training-fold narratives only. Every system uses the same lexicon.

## Results

| System | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | Exact match |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 (baseline) | 0.71 | 0.51 | 0.59 | 0.50 | 0.43 | 0.43 | 0.51 |
| v2 | 0.68 | 0.58 | 0.62 | 0.50 | 0.44 | 0.44 | 0.58 |
| **v3 (final)** | **0.77** | **0.71** | **0.74** | **0.52** | **0.47** | **0.47** | **0.71** |
| Keyword lookup | 0.85 | 0.85 | 0.85 | 0.65 | 0.63 | 0.62 | 0.85 |
| Majority class (floor) | 0.12 | 0.12 | 0.12 | 0.00 | 0.00 | 0.00 | 0.12 |

- **Micro** pools all cases. Precision = correct / answered; recall = correct / all 3,000.
- **Macro** averages per-class scores over the 320 product classes in the test set.
- **Exact match** = share of all cases where the predicted code equals `Product_1`. With one gold code per case, this equals micro recall.

**v3 vs the baseline (paired bootstrap, 95% intervals):**

| | Micro F1 | Macro F1 | Exact match |
|---|---|---|---|
| v3 − v1 | +0.148 (0.134 to 0.163) | +0.037 (0.022 to 0.052) | +0.206 (0.190 to 0.222) |
| v3 − v2 | +0.117 (0.103 to 0.130) | +0.035 (0.020 to 0.050) | +0.137 (0.123 to 0.151) |

## Findings

- **v3 answers more often without losing precision.** It answers 93% of cases against v1's 71%, and its precision is higher (0.77 vs 0.71). v3 gets 668 cases right that v1 misses; v1 gets 50 right that v3 misses.
- **The gain is concentrated in common products.** Macro F1 rises only 0.04, because the v3 changes targeted floors, stairs, beds and sports. On the 29 classes with at least 20 test cases, macro F1 rises from 0.62 to 0.76.
- **Workplace-like products barely improved.** On tools, ladders, doors, glass and machinery (210 cases), micro F1 went from 0.69 to 0.70.
- **The keyword lookup scores higher, but it does not extract objects.** It is trained on the same NEISS codes it is scored against and goes straight to a code. Its outputs are cue phrases ("FALL TO", "OUT BED"), not objects, so it does not do the extraction task.

## Error analysis

v3 made 863 errors:

| Error type | Cases | Share |
|---|---:|---:|
| Span maps to a different product | 347 | 40% |
| No answer (OTHER) | 217 | 25% |
| Matches a secondary NEISS product (`Product_2`/`Product_3`) | 191 | 22% |
| Floor or keyword fallback rule was wrong | 78 | 9% |
| Span has no lexicon entry | 30 | 3% |

The most common confusions involve floors (monkey bars → floor, floor → bed, stairs → floor) and sibling codes (unspecified scooter → powered scooter).

A random sample of 50 errors was reviewed by hand:

| Cause | Cases | Examples |
|---|---:|---|
| Missed a named object | 12 | earring, door, bleach, slide, stove |
| Chose the wrong span | 10 | "LIP LACERATION" instead of electric scooter; "CEMENT" instead of wagon |
| Right object, sibling code or mapping | 13 | scooter: powered vs unspecified; "ball" mapped to soccer |
| Several products named, coder picked another | 9 | chair and table; floor and bucket |
| Object not in the text | 6 | redacted object; knife implied by "cutting a cucumber" |

22 of the 50 (44%) are real model errors. Another 22 (44%) are scoring artifacts or defensible answers, and 6 (12%) cannot be solved from the text. So strict code matching understates how often v3 names an acceptable object.

## Limitations

- **The ground truth is a proxy.** NEISS codes record the coder's chosen product, not the object named in the text. No hand labels were collected for the test set, so object-level accuracy is not measured.
- **Code-level scores depend on the lexicon.** Mapping errors count against v3, and against v1 and v2 equally.
- **Macro F1 rests on thin classes.** 291 of the 320 classes have fewer than 20 test cases.
- **One object per case.** v3 predicts a single object, but 17.5% of test cases have more than one product code.

Full tables: `reports/final_eval_results.md`. Per-case results: `reports/final_eval_details.csv`.
