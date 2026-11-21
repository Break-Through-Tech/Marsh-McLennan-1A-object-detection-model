# v4: speed and accuracy changes, validation and test results

## Held-out test (3,000 narratives, 11 test-fold hospitals, `python -m src.final_eval`)

Run once, after the team chose the classifier on validation results. v3 reproduces its committed numbers exactly.

| System | Micro P | Micro R | Micro F1 | Macro F1 | Exact match | Set match |
|---|---:|---:|---:|---:|---:|---:|
| v3 (Task #2) | 0.768 | 0.712 | 0.739 | 0.471 | 0.712 | 0.776 |
| v4 GLiNER + rules | 0.769 | 0.722 | 0.744 | 0.474 | 0.722 | 0.786 |
| Keyword lookup | 0.851 | 0.851 | 0.851 | 0.623 | 0.851 | 0.907 |
| **v4 classifier** | **0.883** | **0.883** | **0.883** | **0.711** | **0.883** | **0.941** |

Classifier minus v3, paired bootstrap 95% intervals: micro F1 +0.144 (0.130 to 0.159), macro F1 +0.240 (0.213 to 0.268), exact match +0.171 (0.156 to 0.186).

On the 210 workplace-like products (tools, ladders, doors, glass, machinery), micro F1 was 0.698 for v3, 0.702 for v4 GLiNER and 0.848 for the classifier.

Full tables: `reports/final_eval_results.md`.

## Validation

All accuracy numbers come from 5,000 narratives in the 10 validation-fold hospitals (`reports/split_by_psu.csv`), scored against the NEISS product code with the same lexicon and metrics as `src/final_eval.py`. The 3,000-case test set was not used. Reproduce with `python -m src.validate_v4`.

## Results

| System | Answered | Micro P | Micro R | Micro F1 | Macro F1 | Set match |
|---|---:|---:|---:|---:|---:|---:|
| v3 (Task #2) | 4,746 | 0.786 | 0.746 | 0.765 | 0.529 | 0.810 |
| v4 GLiNER + rules | 4,783 | 0.789 | 0.755 | 0.772 | 0.528 | 0.819 |
| Keyword lookup | 5,000 | 0.853 | 0.853 | 0.853 | 0.592 | 0.907 |
| Lookup if confident (purity ≥ 0.7), else GLiNER, else lookup | 5,000 | 0.857 | 0.857 | 0.857 | 0.615 | 0.905 |
| **v4 classifier** | 5,000 | **0.898** | **0.898** | **0.898** | **0.737** | **0.949** |

Set match = the predicted code is any of the case's NEISS product codes (Product_1 to 3).

## What changed

**Speed (`src/extract.py`)**
- GLiNER's `inference()` gets the real batch size. The old call, `batch_predict_entities()`, is deprecated and processed 8 texts at a time whatever `--batch-size` said.
- Texts are sorted by length before batching, and duplicate texts run once.
- Together: 1.5x faster on CPU with identical output (600 validation narratives, 2 cores: 13.5 to 20.4 rows/s). Duplicates are only 1.2% of the data (405,331 unique event texts of 410,201), so most of the gain is from batching and sorting.
- `--shard K/N` splits a run across machines, and `--merge` combines the pieces.
- Output is saved in chunks (`--chunk-size`), and a rerun skips finished chunks, so an interrupted run resumes.
- `--device auto` uses a GPU when available. `notebooks/v4_colab_full_run.ipynb` runs the whole thing on a Colab GPU and saves to Google Drive.

**Accuracy**
- `src/classify.py`: TF-IDF word and word-pair features plus a linear model, trained on the 286,946 training-fold narratives (about 5 minutes on a laptop CPU). Predicting codes is fast compared with GLiNER, so it adds little to a full run. GLiNER still names the object, and the output's `agree` column shows where the two point to the same product.
- Activity keywords (exercise, swimming, dancing, weight lifting): these were 53 of the 255 validation cases v3 left as OTHER. The rule fired on 47 cases, and 44 were correct.
- Span rules: drop ground and building materials, drop spans ending in an injury word ("LIP LACERATION"), and drop a vehicle only when the text says it hit the patient or the patient was on another conveyance. These were neutral on NEISS validation. They are kept because in workplace text a truck or forklift is often the object.
- `src/final_eval.py`: v1, v2 and v3 are frozen at their original rules, and still reproduce their reported counts on the 25 development cases. `v4` and `classifier` are added as systems.

**Housekeeping**
- A GitHub Actions workflow runs the tests on every PR.
- `requirements.txt` pins `gliner==0.2.30`, removes spaCy (nothing imported it), and adds scikit-learn, tabulate and pytest.
- `.gitignore` covers model weights again.
- Default outputs point to v4 files.
- `verify_task2.py` is removed. Its check (v3 reproduces 17/3/8 on the 25 cases) is now part of `src/final_eval.py`.

## Tested and not adopted

- **Small open LLM as a fallback.** Qwen2.5-1.5B-Instruct, local, few-shot, was tried on 120 validation cases where the lookup was not confident and GLiNER answered. It scored 0.650, against 0.708 for GLiNER and 0.700 for the lookup, at 1.4 s per case on CPU. A larger model on a GPU is untested.
- **int8 quantization.** Not tried: GLiNER's documentation says stock DeBERTa-based models lose accuracy with it.

## Limits

- **NEISS codes are a proxy for the object named in the text.** See `reports/final_evaluation.md`.
- **The classifier learned NEISS wording.** How it transfers to Marsh's workers' compensation text is unknown. It needs sample narratives from Marsh, and the `agree` column flags cases to check.
- **Ground truth is the NEISS code on both validation and test.** No hand labels were collected for the test set.
