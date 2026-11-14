# Task #2: Model refinement (v3)

**Model:** `urchade/gliner_medium-v2.1` (CPU).
**Hand-labeled validation set:** `data/eval/ground_truth_sample_1.csv` (25 narratives; the same sample used in Tasks #7 and #9).
**Implementation:** `src/extract.py`, `src/tune.py`, and `src/check_rules.py`.

## Validation results

| Version | Scoring rule | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Task #7 (v1) | Strict | 5 | 12 | 20 | 0.29 | 0.20 | 0.24 |
| Task #7 (v1) | Lenient | 9 | 8 | 16 | 0.53 | 0.36 | 0.43 |
| Task #9 (v2) | Strict | 12 | 5 | 13 | 0.71 | 0.48 | 0.57 |
| Task #9 (v2) | Lenient | 16 | 1 | 9 | 0.94 | 0.64 | 0.76 |
| **Task #2 (v3)** | **Strict** | **17** | **3** | **8** | **0.85** | **0.68** | **0.76** |
| **Task #2 (v3)** | **Lenient** | **20** | **0** | **5** | **1.00** | **0.80** | **0.89** |

With the selected v3 defaults, strict F1 increased from 0.57 to 0.76 and lenient F1 increased from 0.76 to 0.89. Precision and recall also increased under both scoring rules. These figures are measured on 25 cases and should not be interpreted as an estimate of performance on an independent test set.

Per-case results are recorded in `reports/task2_eval_details.csv`.

## Error analysis and changes

The Task #9 error review identified several recurring problems: sports and activities were sometimes returned as `OTHER`; some spans included action words (`HANDLING KNIFE`, `BASKETBALL GAME`); a misspelling (`WHELLCHAIR`) prevented an exact match; and the fallback assigned `FLOOR` when a narrative explicitly named another object such as a bed, steps, shower, scooter, bicycle, or dresser.

The v3 extraction pipeline makes four changes:

1. **Prompt labels:** added `sport` to the GLiNER label set. The labeling instructions count a sport or activity as the object even when no piece of equipment is named.
2. **Span cleanup:** removes selected leading action/side words and trailing activity words before the stoplist and scorer see a span. It also corrects high-confidence near-miss spellings from a limited object vocabulary. Examples include `HANDLING KNIFE` → `KNIFE`, `BASKETBALL GAME` → `BASKETBALL`, `RIGHT KNEE` → `KNEE`, and `WHELLCHAIR` → `WHEELCHAIR`. Protected product phrases such as `VIDEO GAME` are retained.
3. **Keyword fallback:** when no model span survives, checks for a small list of named objects (bed, stairs/steps, shower, bathtub, scooter, bicycle, and dresser) before applying the existing floor fallback. The model prediction still takes precedence when a span survives.
4. **Tuning and diagnostics:** added `src/tune.py` to compare label sets, text variants, thresholds, span cleanup, and fallback choices; added `src/check_rules.py` to compare rule-generated outputs with the NEISS primary product label.

## Tuning results and selected defaults

The tuning grid evaluated 256 configurations: four label sets, four text/case variants, four thresholds, span cleanup on/off, and floor-only versus keyword-plus-floor fallback. Predictions were cached so the model was run once per label-set/text variant, while thresholds and rule variants were evaluated against the cached predictions. The model weights were **not** fine-tuned: “tuning” here refers to choosing zero-shot labels, input preprocessing, inference thresholds, and post-processing rules.

The highest-scoring grid row used the `sport` label, normalized text, threshold 0.20, span cleanup, and the floor fallback. It scored strict precision 0.83, recall 0.76, F1 0.79; lenient F1 was 0.92. Four of 256 configurations were within 0.02 strict F1 of that row; some of these rows differ only in fallback rules that gave identical predictions on the 25 examples. This alone is not evidence that a particular configuration will generalize better.

The selected production defaults are deliberately different: `sport` label, event text, no lowercasing, threshold 0.30, span cleanup enabled, and keyword-plus-floor fallback. This configuration scored strict F1 0.76 and lenient F1 0.89 on the final direct validation run. Event text was retained because the project logic plan excludes the diagnosis section, which can contain misleading object-like terms. The original 0.30 threshold was retained rather than adopting the 0.20 threshold selected by the tuning grid.

The baseline row in the tuning grid reproduces the Task #9 setup: strict TP 12, FP 5, FN 13 (F1 0.57). The direct evaluation with the selected defaults returned strict TP 17, FP 3, FN 8 and lenient TP 20, FP 0, FN 5. In other words, the selected v3 defaults correctly extracted **five more objects** by the strict rule than v2 on these 25 cases.

**Case-level changes for the highest-scoring grid configuration (not the final default):** it corrected eight former strict misses: basketball game → basketball (`251119278`), ice skating (`250158161`), football (`250916295`), moped (`250455175`), WHELLCHAIR → WHEELCHAIR (`250253877`), handling knife → knife (`250415866`), hockey (`250930046`), and bicycle (`250908300`). It also caused one former strict match to fail: floor → concrete floor (`250712995`), while the gold label was `floor (concrete)`. These comparisons describe the **grid winner**, which uses normalized text and threshold 0.20; they should not be presented as an exact change list for the chosen event-text/0.30 configuration. The exact chosen-default case results are saved in `reports/task2_eval_details.csv`.

## Rule diagnostic on 300 narratives

The v3 pipeline was run on a reproducible 300-narrative sample (seed 100). `src/check_rules.py` reported:

| Rule-generated answer | Fired | Matches NEISS primary product label | Agreement |
|---|---:|---:|---:|
| Floor | 18 | 13 | 72.2% |
| Shower | 6 | 6 | 100.0% |
| Bed | 5 | 5 | 100.0% |
| Stairs | 4 | 4 | 100.0% |
| Bicycle | 2 | 2 | 100.0% |
| Bathtub | 1 | 1 | 100.0% |
| Dresser | 1 | 1 | 100.0% |
| **All rule-generated predictions** | **37** | **32** | **86.5%** |

The run produced 14 `OTHER` predictions (4.7% of the sample). For comparison, the earlier Task #9 check reported 38 floor-rule predictions, 15 matches (39.5%), and 29 `OTHER` predictions.

**Important limitation:** the keyword list was created after reviewing 23 disagreements from the earlier Task #9 run on this same 300-row sample. Therefore, the keyword rule's 19/19 agreement is an in-sample diagnostic, not an independent validation result. The new result shows that the rule matches the cases it was designed around; it does not establish that it will achieve 100% agreement on new narratives. The comparison with Task #9 is also not a controlled experiment because the label set, span cleanup, and fallback behavior changed together.

## Limitations

- The same 25 hand-labeled cases were used in Tasks #7 and #9 and were reused to tune and select v3. Some of the measured improvement is therefore likely fitted to this small validation set. A new held-out set of freshly labeled narratives is required for a credible generalization estimate.
- The 300-row keyword diagnostic is not independent because the keyword list was derived from disagreements in that sample. NEISS's primary product code is also an imperfect proxy for the single object selected from a narrative; agreement is a rough diagnostic, not a gold-standard accuracy measure.
- Strict scoring penalizes wording differences. For example, `CONCRETE FLOOR` versus the gold label `floor (concrete)` can fail strict matching while passing lenient matching. A taxonomy-aware span-to-code mapping could reduce this scoring artifact.
- A keyword appearing in a narrative is not necessarily the object that caused the injury. The keyword fallback is intentionally limited and should be reviewed against a fresh labeled set before being treated as a general solution.
- The earlier Task #10 full run covered 410,201 narratives using the Task #9 rules. The full dataset has not been rerun with v3; the earlier CPU run took about six hours.

## Next steps

1. Evaluate v3 on a fresh held-out set that was not used to choose labels, thresholds, or rules.
2. Review keyword-fallback errors on that held-out set and expand or remove patterns based on evidence.
3. Add a taxonomy-aware span-to-code mapping layer and report exact-match and set-based metrics separately.
4. Rerun the full 410,201-row pipeline with v3 only when the team has agreed that the validation and rule checks justify the additional CPU runtime.
