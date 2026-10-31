# Task #7: Evaluation of extraction pipeline v1

Ground truth: `data/eval/ground_truth_sample_1.csv` (25 hand-labeled narratives, Task #6).
Pipeline settings: model `urchade/gliner_medium-v2.1`, labels ['object', 'tool', 'furniture', 'sports equipment', 'structure'], threshold 0.3, text field `event`

## Metrics

| subset | rule | n | answered | TP | FP | FN | precision | recall | f1 |
|---|---|---|---|---|---|---|---|---|---|
| all | strict | 25 | 17 | 5 | 12 | 20 | 0.29 | 0.20 | 0.24 |
| all | lenient | 25 | 17 | 9 | 8 | 16 | 0.53 | 0.36 | 0.43 |
| mention_type=explicit | strict | 22 | 16 | 5 | 11 | 17 | 0.31 | 0.23 | 0.26 |
| mention_type=explicit | lenient | 22 | 16 | 8 | 8 | 14 | 0.50 | 0.36 | 0.42 |
| mention_type=implicit | strict | 2 | 1 | 0 | 1 | 2 | 0.00 | 0.00 | 0.00 |
| mention_type=implicit | lenient | 2 | 1 | 1 | 0 | 1 | 1.00 | 0.50 | 0.67 |
| mention_type=redacted | strict | 1 | 0 | 0 | 0 | 1 | 0.00 | 0.00 | 0.00 |
| mention_type=redacted | lenient | 1 | 0 | 0 | 0 | 1 | 0.00 | 0.00 | 0.00 |

Strict = the extracted span is the gold object name. Lenient = they share a content word (e.g. "tile floor" vs "floor (tile)"). See `src/evaluate.py` for the exact rules.

## Results by case

result
no prediction (OTHER)    8
wrong object             8
correct                  5
partial match            4

## Cases to review for error analysis (Task #8): 20

- `251119278` predicted **BASKETBALL GAME**, gold **basketball (activity)** (explicit): partial match
- `250915721` predicted **STROLLER**, gold **baby stroller** (explicit): partial match
- `251133456` predicted **OTHER**, gold **tea tree oil (lice treatment)** (explicit): no prediction (OTHER)
- `250535303` predicted **OTHER**, gold **footwear (sibling's shoes)** (redacted): no prediction (OTHER)
- `250450072` predicted **OTHER**, gold **string** (explicit): no prediction (OTHER)
- `251225330` predicted **OTHER**, gold **floor** (implicit): no prediction (OTHER)
- `250438619` predicted **PROSTHETIC DEVICE**, gold **chair** (explicit): wrong object
- `250158161` predicted **POND**, gold **ice skating (activity)** (explicit): wrong object
- `250916295` predicted **OTHER**, gold **football (activity)** (explicit): no prediction (OTHER)
- `250455175` predicted **INTERSECTION**, gold **moped** (explicit): wrong object
- `250253877` predicted **WHELLCHAIR**, gold **wheelchair** (explicit): wrong object
- `250415866` predicted **HANDLING KNIFE**, gold **knife** (explicit): partial match
- `250534756` predicted **TAILBONE**, gold **floor** (explicit): wrong object
- `250712995` predicted **OTHER**, gold **floor (concrete)** (explicit): no prediction (OTHER)
- `250930046` predicted **OTHER**, gold **hockey (activity)** (explicit): no prediction (OTHER)
- `250801253` predicted **OTHER**, gold **floor (tile)** (explicit): no prediction (OTHER)
- `250908300` predicted **CAR**, gold **bicycle** (explicit): wrong object
- `250220044` predicted **CAR**, gold **electric scooter** (explicit): wrong object
- `251215704` predicted **NOODLES**, gold **noodle cup/container** (implicit): partial match
- `250617584` predicted **BATHROOM**, gold **floor** (explicit): wrong object

Full per-case output: `reports/task7_eval_details.csv`
