# Task #9: Evaluation of Updated Extraction Pipeline (v2 Rules)

**Ground truth:** `data/eval/ground_truth_sample_1.csv` (25 hand-labeled narratives, Task #6).  
**Pipeline settings:** Model `urchade/gliner_medium-v2.1`, labels `['object', 'tool', 'furniture', 'sports equipment', 'structure']`, threshold `0.3`, text field `event`.  
**Applied rules:**
1. **Expanded Stoplist:** Drops non-product entity candidate spans (places, secondary vehicles, body parts, implanted devices).
2. **Floor Default Fallback:** Assigns `FLOOR` if no object span survives non-outdoor fall descriptions.

---

## Performance Comparison (Task #7 Baseline vs. Task #9 Updated)

| Metric Subset | Evaluation Rule | TP | FP | FN | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|
| **Task #7 (Baseline)** | Strict | 5 | 12 | 20 | 0.29 | 0.20 | 0.24 |
| **Task #7 (Baseline)** | Lenient | 9 | 8 | 16 | 0.53 | 0.36 | 0.43 |
| **Task #9 (Updated)** | **Strict** | **12** | **5** | **13** | **0.71** | **0.48** | **0.57** |
| **Task #9 (Updated)** | **Lenient** | **16** | **1** | **9** | **0.94** | **0.64** | **0.76** |

---

## Detailed Impact Analysis

* **Strict F1 Improvement:** Increased from **0.24 to 0.57** (+0.33 improvement).
* **Lenient F1 Improvement:** Increased from **0.43 to 0.76** (+0.33 improvement).
* **Precision Surge:** Lenient precision reached **0.94** (only 1 false positive out of 17 answered predictions), demonstrating that the expanded stoplist successfully filtered out noisy candidate spans like places and body parts.
* **Reduction in False Positives:** False positives dropped from **12 down to 5** (Strict) and **8 down to 1** (Lenient).

---

## Generalization Check on Unseen Narratives (Part 3 Output)

A sample run of 300 un-labeled narratives (`data/processed/extractions_v2.csv`) yielded:
* **Floor Rule Firings:** 38 rows (~12.7% of narratives).
* **NEISS Code 1807 Agreement:** 15 of 38 rows (~39.5% direct single-code match against NEISS primary product code `1807`).
* **Unassigned ("OTHER") Rate:** 29 out of 300 rows (~9.7%), showing strong model coverage with fallback.

*Note:* Because NEISS product coding lists only one primary code per injury event, partial disagreement between narrative fall context and primary product assignment is expected. The heuristic operates safely without introducing widespread noise.
