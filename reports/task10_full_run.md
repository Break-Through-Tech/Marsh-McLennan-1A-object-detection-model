# Task #10: Full Pipeline Run (410,201 narratives)

**Status: complete.** The refined v2 pipeline ran on all 410,201 cleaned NEISS narratives with 0 failed rows. Figures come from `reports/task10_full_run.json` and `reports/task10_smoke_test.json`.

## Configuration

| Setting | Value |
|---|---|
| Model | `urchade/gliner_medium-v2.1` |
| Labels | object, tool, furniture, sports equipment, structure |
| Threshold | 0.3 |
| Text field | `event` (narrative without the DX section) |
| Batch size | 32 |
| Rules | Task #9 v2: expanded stoplist, floor default for falls (unchanged) |
| Environment | Python 3.12.13, CPU only (torch 2.15.0+cpu) |

## Input build

`data/raw/neiss2025.xlsx` (sheets NEISS_2025: 410,201 x 25; NEISS_FMT: 1,250 x 4) was converted to CSV per `data/README.md`. `notebooks/milestone1_eda.ipynb` was executed (a copy, original unchanged) and wrote `data/processed/neiss2025_clean.csv`: 410,201 rows x 24 columns, `narrative` has no nulls, `case_number` is unique.

## Commands

```
python -m src.extract --input data/processed/neiss2025_clean.csv --output data/processed/extractions_v2_sample.csv --sample 1000 --text-field event --threshold 0.3 --batch-size 32 --log reports/task10_smoke_test.json
python -m src.extract --input data/processed/neiss2025_clean.csv --output data/processed/extractions_v2.csv --text-field event --threshold 0.3 --batch-size 32 --log reports/task10_full_run.json
```

## Smoke test (1,000 random rows, seed 100)

1,000 rows out, 0 failed, 62.3 s (16.06 rows/s), OTHER 123 (12.3%), floor rule 130. Output has `case_number` and `extracted_object` with no nulls. `extracted_entity_type` is null only on OTHER rows and `extraction_confidence` only on OTHER and floor-rule rows, by design (no model span). Projected full runtime was about 7.1 h.

## Full run

| Item | Value |
|---|---|
| Input rows | 410,201 |
| Output rows | 410,201 (`case_number` unique) |
| Runtime | 21,448 s (about 5.96 h), 19.13 rows/s |
| OTHER (fallback) | 52,436 (12.78%) |
| Floor-rule predictions | 56,045 (13.66%) |
| Failed / missing rows | 0 |

Top extracted objects (count): FLOOR 61,076; OTHER 52,436; BED 13,804; STAIRS 10,747; CHAIR 9,524; WALL 8,353; COUCH 6,014; BASKETBALL 5,625; KNIFE 5,143; DOOR 5,025; TABLE 4,374; STEPS 3,890; TOILET 2,712; HELMET 2,687; BALL 2,669; TRAMPOLINE 2,328; BICYCLE 2,209; LADDER 2,157; SCOOTER 2,080; FOOTBALL 2,075. FLOOR counts include rows from the model as well as the floor rule (61,076 vs 56,045 rule rows).

## Code change

`src/extract.py` gained `summarize_run()` and a `--log PATH` option (JSON: counts, runtime, rows/s, OTHER and floor-rule counts, failed rows, top 20 objects, config). Two tests were added. The v2 rules are unchanged.

## Limitations

- No ground truth exists for the full set. The only accuracy evidence is the Task #9 25-case sample (strict F1 0.57, lenient F1 0.76); full-run quality is not measured here.
- "Failed" means a missing or blank `extracted_object`. It does not detect wrong answers.
- 12.78% OTHER includes redacted narratives where the object may be hidden by `*`.
- `data/processed/extractions_v2.csv` (and the other data files) are gitignored and not committed. It is regenerable with the command above.
- Run on CPU with `batch_predict_entities`, which emits a deprecation warning in the installed GLiNER.
