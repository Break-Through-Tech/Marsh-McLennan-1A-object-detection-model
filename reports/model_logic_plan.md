# Model Logic Plan

Marsh McLennan 1A, Break Through Tech AI Studio, Fall 2026
September milestone, Task #12

This plan consolidates the EDA findings (`reports/eda_report.md`, `notebooks/milestone1_eda.ipynb`, Task #6 preprocessing notebook) and the chosen NLP/LLM approach. It is the basis for the October v1 build.

---

## 1. Goal

Given a short free-text injury narrative, extract the object involved and map it to a NEISS product code.

- **Train and tune data:** NEISS 2025 (410,201 ED visits, 25 columns).
- **Final test data:** Marsh's anonymized workers' compensation descriptions, which have no NEISS codes. Marsh will compare our output with its existing object detection model.
- **Input:** `Narrative_1` only. **Label:** `Product_1`.

## 2. What the EDA tells us, and what we do about it

| Finding | Design decision |
|---|---|
| Median narrative is 114 characters (21 words). | Context length is not a concern. Cost depends on case count, so batch narratives when calling an LLM. |
| 99.81% start with an age/sex token (`47YOF`, but also `59 YOF` with a space and `15MOF` for months). 99.9% contain a `DX` marker, and only 74.4% write it with a colon. The segment after `DX` describes the injury, not the object, and holds bait words such as FALL and FOREIGN BODY. | Split each narrative into prefix, event clause, and diagnosis. Feed the model the event clause only. |
| 100% of narratives are uppercase. Most pretrained NER models degrade on all-caps text. | Test case normalization of the event clause as a preprocessing option and keep whichever scores higher on validation. |
| Shorthand such as `C/O` and `PT` is common. Some words are redacted with `***` (3.2%). | Expand `C/O` and `PT` (verified in Task #6). Leave other abbreviations unchanged. Keep the raw narrative column untouched. |
| `Product_1` has 767 classes. The top 157 cover 90% of records. 250 classes have fewer than 20 cases (0.44% of rows). | Merge classes under 20 cases into `OTHER`. Report macro-F1 with and without the tail. |
| 22% of labels are vague (for example NOT SPECIFIED, N.E.C.). | Do not use vague-label rows to build the lexicon or tune thresholds. Report them as a separate evaluation slice. |
| The product name appears in the narrative for 66% of records (72% excluding vague labels). | Extraction can solve about two thirds of cases directly. The rest need inference (for example "chopping cabbage" means knife). |
| 82.8% of cases have exactly one product code. Slot order is inconsistent between coders (12,881 ordered pairs vs 10,014 unordered). | v1 is single-object. Score predictions as a set, not by slot position. Multi-label is a refinement. |
| 77 hospitals write in different styles. Some always use `DX:`, others never do. | Split by hospital (`PSU`). Do not use `PSU` as a feature. |
| Products are seasonal (swimming peaks at 4.1x). | No chronological split. |
| 0.45% of narratives are duplicated, and 7 duplicate texts carry conflicting labels. | Deduplicate across fold boundaries. Accept a small ceiling on achievable accuracy. |

## 3. Pipeline

```
Narrative_1
  -> 1. Preprocess
  -> 2. GLiNER extraction (zero-shot)
  -> 3. spaCy / rule-based cleanup
  -> 4. Map span to NEISS product code
  -> 5. Fallback for unmapped cases
  -> predicted set of product codes
```

1. **Preprocess.** Collapse whitespace, expand `C/O` and `PT`, split off the age/sex prefix (allowing a space and month-old forms) and the diagnosis segment (match `DX` with or without a colon), and keep the event clause. Optionally normalize case.
2. **Extract.** Run GLiNER on the event clause with a small set of object labels (for example object, tool, furniture, sports equipment, structure). The label set and the score threshold are tuned on the validation fold. GLiNER needs no labeled training set and runs locally.
3. **Clean up.** Use spaCy to lemmatize spans and handle plurals. Rules drop spans that are body parts, people, or injury words.
4. **Map to a NEISS code.** The NEISS taxonomy (`NEISS_FMT`, 1,124 codes) uses wording that clinicians do not type (FOOTWEAR vs "shoe", JEWELRY vs "ring"). Mapping uses two layers:
   - A lexicon built from taxonomy labels plus curated synonyms, matched on the lemma.
   - Embedding similarity between the span and taxonomy labels, accepted above a tuned threshold.
   Several spans in one narrative produce several codes.
5. **Fallback.** If no span survives or no code maps above the threshold, v1 outputs `OTHER`. Section 4 describes the LLM fallback that is evaluated in October.

## 4. Role of the LLM

The project brief recommends LLM techniques, and the EDA shows why: about a third of labels are not named in the text, and no lexicon can solve those.

- **v1 (October):** GLiNER + spaCy pipeline only. No API cost.
- **LLM comparison:** run an LLM through the OpenAI SDK on the same validation sample (each team member supplies their own API key). Select few-shot exemplars stratified across product groups from the training fold only, so the model does not anchor on floors and beds. Batch many narratives per request.
- **Decision rule:** adopt the LLM as the fallback for step 5 if it improves validation macro-F1 by enough to justify the cost. The team sets the margin after seeing v1 results.

## 5. Data splits

- **Primary split:** whole hospitals assigned to folds, largest hospital first, to the fold furthest below target. Result: train 56 hospitals (69.95%), validation 10 (15.01%), test 11 (15.04%).
- **Control split:** a stratified random split on `Product_1`, used with the same model. The stratified score is the optimistic ceiling. The hospital-split score is the realistic estimate for Marsh's data. The gap between them shows how much the model relies on house style.
- **Use of folds:** the training fold supplies lexicon synonyms and few-shot exemplars. The validation fold sets thresholds and preprocessing choices. The test fold is held back for final evaluation.
- **Known cost of the hospital split:** venue mix drifts between folds (Home differs by 14 percentage points). We report this alongside results.
- **Workplace-like slice:** a validation subset of tool, machinery, and structural codes (ladder, scaffold, nail gun, welding, saws, grinders, and similar), reported separately.

## 6. Evaluation

| Metric | How it is computed |
|---|---|
| Hit rate (primary) | `Product_1` appears in the predicted set. |
| Precision, recall, F1 | Micro and macro. Macro is reported over all classes and over classes with 20 or more cases. |
| Exact match accuracy | Predicted set equals the gold product set (`Product_1` to `Product_3`, non-zero codes). |
| Slice reports | Vague-label rows, workplace-like slice, hospital split vs stratified control. |

Exact match against `Product_1` undercounts correct extractions, because coders sometimes pick a code whose name is not in the text. We review a sample of misses by hand and report the share that are coder-choice cases.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Domain gap. NEISS is a consumer product system. Only 0.01% of cases are coded Industrial. | Workplace-like validation slice, oversampling of tool and machinery codes, pooling several NEISS years. Ask the Challenge Advisor for NEISS-Work, the stretch goal and highest-value option. |
| Some workplace objects have a code but no 2025 examples (fork lift, conveyor). PALLET and CRANE have no code. | Cover through lexicon synonyms and the LLM fallback. Flag as a known limit in results. |
| Vocabulary mismatch between narratives and taxonomy. | Curated lexicon plus embedding lookup (step 4). |
| Genuine inference cases, where the object is never named. | LLM fallback (section 4). |
| Coder disagreement caps accuracy (for example BED vs FLOOR). | Set-based scoring. State the ceiling when reporting results. |
| All-caps text lowers GLiNER quality. | Case normalization tested on validation. |

## 8. October deliverables

- Preprocessing functions merged into `src/` (from the Task #6 notebook).
- GLiNER + spaCy pipeline with the lexicon mapping, run on a validation sample.
- First metrics table (section 6), including the hospital-split vs stratified-control comparison.
- LLM baseline on the same sample.
- Updated README results section.

## 9. Open decisions

- Whether to adopt the LLM fallback (decided after v1 results).
- Whether `Location`, `Product_2`, and `Product_3` are used later (currently undecided).
- NEISS-Work access (request to Challenge Advisor Kamraan Kamal).
