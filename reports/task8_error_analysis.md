# Task #8: Error analysis of extraction pipeline v1

**Data:** the 25 hand-labeled narratives from Task #6 (`data/eval/ground_truth_sample_1.csv`), scored in Task #7 (`reports/task7_eval_details.csv`). The pipeline got 5 right and missed 20: 8 returned `OTHER`, 8 returned a wrong object, and 4 returned a partial match.
**Pipeline:** `urchade/gliner_medium-v2.1`, labels object / tool / furniture / sports equipment / structure, threshold 0.30, event text only.
**Method:** each of the 20 misses was read and given one primary pattern. The full list is in `reports/task8_error_patterns.csv`.
**Caveat:** 25 cases is a small sample, so read the counts as a ranking and not as rates.

## Ranked patterns

| Rank | Pattern | Cases | Share of misses | What goes wrong | Example |
|---|---|---|---|---|---|
| 1 | Floor or surface is the gold object | 5 | 25% | Model returns nothing, a room, or a body part. Labeling rule 4 makes floor the default for falls with no other product. | `250617584`: gold floor, model returned BATHROOM |
| 2 | Wrong object chosen from several in the narrative (multiple objects) | 4 | 20% | The narrative names several things, and the model picks a car, a place, or an implanted device instead of the product. | `250908300`: gold bicycle, model returned CAR |
| 3 (tie) | Activity or sport is the gold object | 3 | 15% | Sports count as the object (rule 3), but no equipment is named. Model returns nothing or a place. | `250916295`: gold football, model returned OTHER |
| 3 (tie) | Right object, different wording | 3 | 15% | Span has an extra word (GAME, HANDLING) or lacks a modifier (baby). Mostly a scoring effect. | `250415866`: gold knife, model returned HANDLING KNIFE |
| 5 | Ambiguous or borderline product | 2 | 10% | Gold depends on a coding convention: a substance (tea tree oil) or an unnamed container (noodle cup). | `251215704`: gold noodle cup, model returned NOODLES |
| 6 (tie) | Typo | 1 | 5% | The narrative misspells the object. | `250253877`: WHELLCHAIR |
| 6 (tie) | Redacted object | 1 | 5% | The object is masked as `***`. No model can recover it. | `250535303` |
| 6 (tie) | Small everyday item missed | 1 | 5% | Plain household item scored under the threshold or does not fit the label set. | `250450072`: gold string |
| 9 | Abbreviations | 0 | 0% | See below. | |

**Abbreviations.** 7 of the 20 narratives contain C/O, S/P, PT or W/ in the text the model reads. `src/preprocess.py` expands these before extraction, and in none of the 7 does an abbreviation hide the object. So abbreviations are not an error source in this sample.

## A second view: what the model returned

| Returned | Count |
|---|---|
| Nothing (`OTHER`) | 8 |
| A place (pond, intersection, bathroom) | 3 |
| A vehicle that hit the patient (car) | 2 |
| Right object with extra or missing words | 3 |
| A body part, a medical device, a food, or a misspelling | 4 |

Seven of the 20 misses (place, vehicle, body part, medical device) are spans that are clearly not consumer products. A stoplist rule would remove those wrong answers. Whether they become correct depends on the next-best span being right, so this raises precision more than recall.

## Caveats

- **One primary pattern per case.** Several cases fit two. `250455175` is both multiple objects and place spans. `250534756` fits floor and body part.
- **Some misses come from NEISS coding, not the model.** Wheelchairs (1707) and mopeds (3215) are unused in the 2025 data, so coders used chairs (4074) and 5046. The team needs a mapping rule for these.
- **Eight misses are abstentions.** Checking whether a lower threshold or added labels (activity, floor or surface, household item) recover them is a test for Task #9, not a conclusion here.

## Suggested fixes to test, in the order of the pattern ranking

1. Add a floor default: if a fall is described and no product is found, output floor.
2. Drop spans that are places, body parts, implanted devices, or the other party's vehicle.
3. Add labels for sports or activities and for floors or surfaces, then compare results on the same 25 cases.
4. Trim leading verbs and trailing words such as GAME from spans, or score on a span-to-code mapping.
5. Add spelling correction before extraction.
