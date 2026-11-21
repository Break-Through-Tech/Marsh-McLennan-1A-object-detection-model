"""Final held-out evaluation of the object-extraction model (Milestone 3).

Scores the final model (v4) against earlier versions and baselines on narratives
from hospitals that no earlier task looked at:

    v1        Task #4/#7 pipeline: base labels, people/body-part stoplist only
    v2        Task #9 pipeline: + wider stoplist, + floor default for falls
    v3        Task #2 pipeline: + "sport" label, span cleanup, keyword fallback
    v4        the current src/extract.py defaults: v3 + activity keywords, span rules
    classifier  v4 code step (final): src/classify.py, TF-IDF + linear model trained
              on the training fold. v4 names the object with GLiNER (the `v4`
              system) and assigns the product code with this classifier.
    keyword   model-free baseline: the narrative word most strongly tied to one
              product code in the training fold
    majority  always predicts the most common training-fold product (floors)

Two kinds of ground truth:

    NEISS codes   Product_1 (and Product_2/3) on a random sample of the test fold
                  in reports/split_by_psu.csv. Large and automatic, but a proxy:
                  the coder's code is not always the object named in the text.
    hand labels   optional. `--make-label-sheet` writes a blank 100-case sheet to
                  data/eval/ground_truth_test_1.csv for labeling with
                  data/eval/Labeling_Instructions.md. When that file has
                  gold_object labels, span-level scores are added to the report.

Spans become NEISS codes through a lexicon built from the training fold only
(CodeMapper): each word or word pair maps to the product code it most often
appears with. The same mapper is used for every system, so it adds the same
noise to all of them.

Run from the repo root (needs data/processed/neiss2025_clean.csv, see data/README.md):

    python -m src.final_eval                    # build sets if missing, run GLiNER, score
    python -m src.final_eval --rebuild-test-set # resample the test set (overwrites it)

GLiNER output is cached in data/processed/final_eval_cache/, so a rerun only
rescores.

Metric definitions (one predicted object per narrative, OTHER = abstain)
    micro    TP = answered and correct, FP = answered and wrong, FN = not correct.
             P = TP / answered, R = TP / n. Same convention as src/evaluate.py.
    macro    per product class c: P_c = TP_c / predicted c, R_c = TP_c / gold c,
             F1_c from those; averaged over the gold classes in the test set
             (and separately over classes with at least 20 test cases).
    exact match (Product_1)   share of all n where the predicted code is Product_1.
             With one gold code per narrative this equals micro recall.
    exact match (product set) share where {predicted code} equals the full set
             of non-zero Product_1..3 codes. A single-object model cannot get a
             multi-product case right under this rule.
    set match                 share where the predicted code is any of Product_1..3.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:  # python -m src.final_eval
    from src import evaluate, extract, tune
except ImportError:  # python src/final_eval.py
    import evaluate
    import extract
    import tune

SPLIT_PATH = Path("reports/split_by_psu.csv")
FMT_PATH = Path("data/raw/neiss2025 1(NEISS_FMT).csv")
TEST_SET_PATH = Path("data/eval/test_set_neiss.csv")
HAND_LABEL_PATH = Path("data/eval/ground_truth_test_1.csv")
CACHE_DIR = Path("data/processed/final_eval_cache")
DETAILS_PATH = Path("reports/final_eval_details.csv")
METRICS_PATH = Path("reports/final_eval_metrics.csv")
SUMMARY_PATH = Path("reports/final_eval_results.md")
# One error sample per system in ERROR_SYSTEMS, for hand review; never overwritten.
ERROR_SAMPLE_DIR = Path("reports")
FIGURE_PATH = Path("reports/figures/final_eval_comparison.png")

TEST_N = 3000
HAND_N = 100
TEST_SEED = 2026
DIAGNOSTIC_SAMPLE = (300, 100)  # (rows, seed) of the Task #9 / Task #2 rule check
LEXICON_MIN_SUPPORT = 20  # a lexicon term must appear in this many training narratives
MACRO_MIN_SUPPORT = 20  # second macro average: classes with this many test cases
BOOTSTRAP_ROUNDS = 1000
ERROR_SAMPLE_N = 50

ABSTAIN = -1  # predicted code when the system returns OTHER
UNMAPPED = -2  # predicted code when the span has no lexicon entry

SYSTEMS = ("v1", "v2", "v3", "v4", "classifier", "keyword", "majority")
FINAL = "classifier"  # v4's product-code step
OBJECT_MODEL = "v4"  # v4's object step (GLiNER + rules)
ERROR_SYSTEMS = (FINAL, "v3")  # error tables and review samples; v3 was the final model before v4
# Paired comparisons for the bootstrap and win/loss tables: the final system against
# every other one, plus the earlier version-to-version steps.
HISTORY_PAIRS = (("v3", "v1"), ("v3", "v2"), ("v4", "v3"))

# ---------------------------------------------------------------- pipelines

# The v1 stoplist from commit 1626d85 (Task #4): people and a few body parts.
V1_DROP_SPANS = frozenset({
    "head", "face", "eye", "eyes", "nose", "mouth", "ear", "neck", "arm", "hand",
    "finger", "thumb", "leg", "knee", "ankle", "foot", "toe", "back", "chest",
    "shoulder", "wrist", "elbow", "hip",
    "brother", "sister", "mother", "father", "friend", "son", "daughter", "patient",
})


def v1_keep_entity(entity: dict) -> bool:
    return entity["text"].strip().lower() not in V1_DROP_SPANS


PIPELINES: Dict[str, extract.ExtractionConfig] = {
    "v1": extract.ExtractionConfig(labels=tune.BASE_LABELS, keep_entity=v1_keep_entity, keep_in_context=None,
                                   clean_span=None, fallback_rule=None),
    "v2": extract.ExtractionConfig(labels=tune.BASE_LABELS, clean_span=None,
                                   keep_entity=extract.v3_keep_entity, keep_in_context=None,
                                   fallback_rule=extract.floor_for_falls),
    "v3": extract.ExtractionConfig(keep_entity=extract.v3_keep_entity, keep_in_context=None,
                                   fallback_rule=extract.v3_keyword_then_floor),
    "v4": extract.ExtractionConfig(),  # the current src/extract.py defaults
}

# Counts each version reported on the 25 development cases (strict TP, FP, FN),
# used to check that this script reproduces the earlier runs.
REPORTED_DEV_COUNTS = {"v1": (5, 12, 20), "v2": (12, 5, 13), "v3": (17, 3, 8), "v4": (17, 3, 8)}

# ---------------------------------------------------------------- data


def load_folds(path: Path = SPLIT_PATH) -> Dict[int, str]:
    split = pd.read_csv(path)
    return dict(zip(split["PSU"].astype(int), split["fold"].astype(str)))


def load_product_names(path: Path = FMT_PATH) -> Dict[int, str]:
    """NEISS product code -> name, from the NEISS_FMT codebook sheet."""
    fmt = pd.read_csv(path)
    fmt.columns = ["format", "code", "end", "label"]
    prod = fmt[fmt["format"].astype(str).str.strip() == "PROD"]
    names = prod["label"].astype(str).str.replace(r"^\d+ - ", "", regex=True)
    return dict(zip(prod["code"].astype(int), names))


def development_case_numbers(clean: pd.DataFrame, dev_gold: pd.DataFrame) -> set:
    """Cases already used to build or tune a model: the 25 hand-labeled cases and
    the 300-row sample the keyword rule was written from."""
    used = set(dev_gold["case_number"].astype(str))
    n, seed = DIAGNOSTIC_SAMPLE
    if len(clean) > n:
        used |= set(clean.sample(n=n, random_state=seed)["case_number"].astype(str))
    return used


TEST_SET_COLUMNS = ["case_number", "narrative", "product_code", "product_label",
                    "product_2_code", "product_3_code", "vague_label", "rare_class",
                    "redacted", "psu"]


def build_test_set(clean: pd.DataFrame, folds: Dict[int, str], exclude: Iterable[str],
                   n: int = TEST_N, seed: int = TEST_SEED) -> pd.DataFrame:
    """Random sample of the test-fold hospitals, minus development cases and any
    narrative that also appears in a train or validation hospital."""
    fold = clean["psu"].astype(int).map(folds)
    exclude = set(map(str, exclude))
    seen_elsewhere = set(clean.loc[fold != "test", "narrative"])
    pool = clean[(fold == "test")
                 & ~clean["case_number"].astype(str).isin(exclude)
                 & ~clean["narrative"].isin(seen_elsewhere)]
    pool = pool.drop_duplicates("narrative")
    sample = pool.sample(n=min(n, len(pool)), random_state=seed)
    out = sample[[c for c in TEST_SET_COLUMNS if c in sample.columns]].copy()
    out["case_number"] = out["case_number"].astype(str)
    return out.sort_values("case_number").reset_index(drop=True)


def hand_label_template(test: pd.DataFrame, n: int = HAND_N, seed: int = TEST_SEED) -> pd.DataFrame:
    """Blank labeling sheet in the same layout as ground_truth_sample_1.csv.
    Model predictions are left out so labelers are not anchored on them."""
    pick = test.sample(n=min(n, len(test)), random_state=seed).sort_values("case_number")
    return pd.DataFrame({
        "case_number": pick["case_number"].values,
        "narrative": pick["narrative"].values,
        "gold_object": "",
        "object_code": "",
        "mention_type": "",
        "neiss_product_1": pick["product_code"].values,
        "matches_neiss": "",
        "notes": "",
    })


def gold_code_set(row) -> frozenset:
    codes = [row.get("product_code"), row.get("product_2_code"), row.get("product_3_code")]
    return frozenset(int(c) for c in codes if pd.notna(c) and int(c) != 0)


# ---------------------------------------------------------------- span -> code lexicon


def terms(text: str) -> set:
    """Words and adjacent word pairs, lowercased and singularized like src/evaluate.py."""
    words = [w for w in evaluate._words(text) if not w.isdigit() and not re.match(r"^\d", w)]
    out = set(words)
    out.update(" ".join(words[i:i + 2]) for i in range(len(words) - 1))
    return out


class CodeMapper:
    """Maps a span to the NEISS product code its words most often appear with."""

    def __init__(self, table: pd.DataFrame):
        # columns: term, code, n, support, purity
        self.table = table.set_index("term")
        self._code = self.table["code"].to_dict()

    @classmethod
    def fit(cls, texts: pd.Series, codes: pd.Series,
            min_support: int = LEXICON_MIN_SUPPORT) -> "CodeMapper":
        rows = pd.DataFrame({"term": [sorted(terms(t)) for t in texts.fillna("")],
                             "code": codes.astype(int).values}).explode("term").dropna()
        counts = rows.groupby(["term", "code"]).size().rename("n").reset_index()
        support = counts.groupby("term")["n"].sum()
        best = (counts.sort_values(["term", "n", "code"], ascending=[True, False, True])
                .drop_duplicates("term").copy())
        best["support"] = best["term"].map(support)
        best["purity"] = best["n"] / best["support"]
        return cls(best[best["support"] >= min_support].reset_index(drop=True))

    def __contains__(self, term: str) -> bool:
        return term in self._code

    def map_span(self, span) -> int:
        """Last word pair, then the head (last) word, then earlier words; first hit wins."""
        if span is None or (isinstance(span, float) and np.isnan(span)):
            return UNMAPPED
        words = [w for w in evaluate._words(span) if not w.isdigit()]
        candidates = []
        if len(words) >= 2:
            candidates.append(" ".join(words[-2:]))
        candidates += list(reversed(words))
        for term in candidates:
            if term in self._code:
                return int(self._code[term])
        return UNMAPPED

    def keyword_predict(self, text: str):
        """Model-free baseline: the term in the text with the purest code (ties: more support)."""
        best = None
        for term in terms(text):
            if term in self._code:
                row = self.table.loc[term]
                key = (float(row["purity"]), int(row["support"]), term)
                if best is None or key > best:
                    best = key
        if best is None:
            return None, ABSTAIN
        return best[2].upper(), int(self._code[best[2]])


# ---------------------------------------------------------------- running the systems


class LazyModel:
    """Loads GLiNER on first use, so a fully cached rerun never loads it."""

    def __init__(self, loader: Callable):
        self.loader = loader
        self.model = None

    def get(self, config):
        if self.model is None:
            print(f"Loading {config.model_name} ...")
            self.model = self.loader(config)
        return self.model


def cache_path(config: extract.ExtractionConfig, cache_dir: Path) -> Path:
    labels = "_".join(l.replace(" ", "-") for l in config.labels)
    name = f"{labels}__{config.text_field}__lc{int(config.lowercase)}__t{config.threshold:.2f}.json"
    return cache_dir / name


def entity_cache(narratives: pd.Series, config: extract.ExtractionConfig,
                 model: LazyModel, cache_dir: Path) -> dict:
    """GLiNER entities for every model text, read from disk where already computed."""
    path = cache_path(config, cache_dir)
    cache = json.loads(path.read_text()) if path.exists() else {}
    texts = extract.prepare_text(narratives, config)["model_text"]
    todo = [t for t in dict.fromkeys(texts) if t and t.strip() and t not in cache]
    if todo:
        print(f"Running GLiNER on {len(todo):,} texts with labels {list(config.labels)} ...")
        found = extract.predict_entities(model.get(config), todo, config, progress=True)
        for t, ents in zip(todo, found):
            cache[t] = [{k: (float(v) if k == "score" else v) for k, v in e.items()} for e in ents]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
    return cache


def run_pipeline(df: pd.DataFrame, config: extract.ExtractionConfig, cache: dict) -> pd.DataFrame:
    model = tune.CachedModel(cache, config.threshold)
    out = extract.run(df[["case_number", "narrative"]], model, config)
    return out[["case_number", "extracted_object", "extracted_entity_type", "extraction_confidence"]]


def predict_all(df: pd.DataFrame, mapper: CodeMapper, majority_code: int, majority_span: str,
                model: LazyModel, cache_dir: Path, classifier=None) -> Dict[str, pd.DataFrame]:
    """One frame per system: case_number, span, entity_type, code."""
    preds = {}
    for name, config in PIPELINES.items():
        cache = entity_cache(df["narrative"], config, model, cache_dir)
        p = run_pipeline(df, config, cache).rename(
            columns={"extracted_object": "span", "extracted_entity_type": "entity_type",
                     "extraction_confidence": "confidence"})
        abstain = p["span"].fillna(extract.FALLBACK).str.upper() == extract.FALLBACK
        p["code"] = [ABSTAIN if a else mapper.map_span(s) for s, a in zip(p["span"], abstain)]
        preds[name] = p

    events = extract.prepare_text(df["narrative"], extract.ExtractionConfig())["event_text"]
    kw = [mapper.keyword_predict(t) for t in events]
    preds["keyword"] = pd.DataFrame({
        "case_number": df["case_number"].values,
        "span": [s if s is not None else extract.FALLBACK for s, _ in kw],
        "entity_type": "baseline: keyword", "confidence": np.nan,
        "code": [c for _, c in kw]})
    if classifier is not None:  # v4: src/classify.py, trained on the training fold
        codes, margins = classifier.predict(events)
        preds["classifier"] = pd.DataFrame({
            "case_number": df["case_number"].values, "span": "", "entity_type": "classifier",
            "confidence": margins, "code": codes})
    preds["majority"] = pd.DataFrame({
        "case_number": df["case_number"].values, "span": majority_span,
        "entity_type": "baseline: majority", "confidence": np.nan, "code": majority_code})
    return preds


# ---------------------------------------------------------------- metrics


def micro(correct: np.ndarray, answered: np.ndarray) -> dict:
    correct = np.asarray(correct, bool)
    answered = np.asarray(answered, bool)
    n = len(correct)
    tp = int((correct & answered).sum())
    fp = int((answered & ~correct).sum())
    fn = n - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n if n else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"n": n, "answered": int(answered.sum()), "TP": tp, "FP": fp, "FN": fn,
            "precision": p, "recall": r, "f1": f1}


def macro(gold: np.ndarray, pred: np.ndarray, classes: Optional[np.ndarray] = None) -> dict:
    """Macro P/R/F1 over `classes` (default: gold classes present). A class with no
    predictions has precision 0, as in scikit-learn's zero_division=0."""
    gold = np.asarray(gold)
    pred = np.asarray(pred)
    if classes is None:
        classes = np.unique(gold)
    classes = np.asarray(classes)
    present = np.isin(classes, gold)
    classes = classes[present]
    if len(classes) == 0:
        return {"classes": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    index = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    g = np.array([index.get(c, k) for c in gold])
    p = np.array([index.get(c, k) for c in pred])
    tp = np.bincount(g[(g == p) & (g < k)], minlength=k)[:k]
    gold_n = np.bincount(g[g < k], minlength=k)[:k]
    pred_n = np.bincount(p[p < k], minlength=k)[:k]
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_n > 0, tp / np.maximum(pred_n, 1), 0.0)
        rec = np.where(gold_n > 0, tp / np.maximum(gold_n, 1), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return {"classes": k, "precision": float(prec.mean()), "recall": float(rec.mean()),
            "f1": float(f1.mean())}


def per_class(gold: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in np.unique(gold):
        tp = int(((gold == c) & (pred == c)).sum())
        g = int((gold == c).sum())
        pr = int((pred == c).sum())
        p = tp / pr if pr else 0.0
        r = tp / g if g else 0.0
        rows.append({"code": int(c), "support": g, "predicted": pr, "TP": tp,
                     "precision": p, "recall": r,
                     "f1": 2 * p * r / (p + r) if p + r else 0.0})
    return pd.DataFrame(rows)


def score_codes(test: pd.DataFrame, preds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide per-case frame: gold fields plus, for each system, span/code/correctness."""
    out = test.copy()
    out["gold_set"] = out.apply(gold_code_set, axis=1)
    for name, p in preds.items():
        p = p.set_index("case_number").reindex(out["case_number"])
        out[f"{name}_span"] = p["span"].values
        out[f"{name}_entity_type"] = p["entity_type"].values
        out[f"{name}_code"] = p["code"].astype(int).values
        code = out[f"{name}_code"]
        out[f"{name}_answered"] = code != ABSTAIN
        out[f"{name}_correct"] = code == out["product_code"].astype(int)
        out[f"{name}_set_match"] = [c in s for c, s in zip(code, out["gold_set"])]
        out[f"{name}_exact_set"] = [s == frozenset({c}) for c, s in zip(code, out["gold_set"])]
    return out


def system_metrics(scored: pd.DataFrame, name: str, frequent: np.ndarray) -> dict:
    gold = scored["product_code"].astype(int).values
    pred = scored[f"{name}_code"].values
    m = micro(scored[f"{name}_correct"].values, scored[f"{name}_answered"].values)
    ms = micro(scored[f"{name}_set_match"].values, scored[f"{name}_answered"].values)
    mac = macro(gold, pred)
    mac_f = macro(gold, pred, frequent)
    return {
        "system": name, "n": m["n"], "answered": m["answered"],
        "abstain_rate": 1 - m["answered"] / m["n"] if m["n"] else 0.0,
        "unmapped_rate": float((pred == UNMAPPED).mean()),
        "TP": m["TP"], "FP": m["FP"], "FN": m["FN"],
        "micro_precision": m["precision"], "micro_recall": m["recall"], "micro_f1": m["f1"],
        "macro_precision": mac["precision"], "macro_recall": mac["recall"], "macro_f1": mac["f1"],
        "macro_classes": mac["classes"],
        f"macro_f1_support{MACRO_MIN_SUPPORT}": mac_f["f1"],
        f"macro_classes_support{MACRO_MIN_SUPPORT}": mac_f["classes"],
        "exact_match_product_1": float(scored[f"{name}_correct"].mean()),
        "exact_match_product_set": float(scored[f"{name}_exact_set"].mean()),
        "set_match": float(scored[f"{name}_set_match"].mean()),
        "set_micro_f1": ms["f1"],
    }


def comparisons(systems, final: str = FINAL, history=HISTORY_PAIRS):
    """(a, b) pairs to compare: final vs every other system, then the history pairs."""
    pairs = [(final, s) for s in systems if s != final and final in systems]
    return pairs + [(a, b) for a, b in history if a in systems and b in systems and (a, b) not in pairs]


def bootstrap(scored: pd.DataFrame, systems, rounds: int = BOOTSTRAP_ROUNDS,
              seed: int = TEST_SEED, pairs=None) -> pd.DataFrame:
    """95% intervals (estimate +/- 1.96 bootstrap SD) for micro F1, macro F1 and exact
    match, and for the paired difference a - b for each (a, b) in `pairs`
    (default: comparisons(systems)).

    Normal intervals rather than percentiles: a resample loses rare classes, which
    mostly score F1 0, so resampled macro F1 drifts above the full-sample value and a
    percentile interval can miss the point estimate entirely.
    """
    rng = np.random.default_rng(seed)
    gold = scored["product_code"].astype(int).values
    classes = np.unique(gold)
    n = len(scored)

    def measure(s, idx):
        correct = scored[f"{s}_correct"].values[idx]
        answered = scored[f"{s}_answered"].values[idx]
        return {"micro_f1": micro(correct, answered)["f1"],
                "macro_f1": macro(gold[idx], scored[f"{s}_code"].values[idx], classes)["f1"],
                "exact_match_product_1": float(correct.mean())}

    point = {s: measure(s, np.arange(n)) for s in systems}
    stats = {s: {m: [] for m in point[s]} for s in systems}
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        for s in systems:
            for metric, value in measure(s, idx).items():
                stats[s][metric].append(value)
    rows = []
    for s in systems:
        for metric, values in stats[s].items():
            est, sd = point[s][metric], float(np.std(values, ddof=1))
            rows.append({"system": s, "metric": metric, "estimate": est,
                         "low": est - 1.96 * sd, "high": est + 1.96 * sd})
    for a, b in (comparisons(systems) if pairs is None else pairs):
        for metric in stats[a]:
            d = np.array(stats[a][metric]) - np.array(stats[b][metric])
            est_d, sd_d = point[a][metric] - point[b][metric], float(np.std(d, ddof=1))
            rows.append({"system": f"{a} - {b}", "metric": metric, "estimate": est_d,
                         "low": est_d - 1.96 * sd_d, "high": est_d + 1.96 * sd_d})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- slices and errors

_LABEL_SKIP = {"OTHER", "NOT", "SPECIFIED", "ELSEWHERE", "CLASSIFIED", "AND", "THE", "WITH",
               "FOR", "EQUIPMENT", "RELATED", "TYPE", "TYPES", "USED", "PARTS", "PART"}

# Tool, machinery and structural products: the closest NEISS gets to workplace objects.
WORKPLACE_PATTERN = re.compile(
    r"LADDER|SCAFFOLD|\bSAWS?\b|CHAIN SAW|DRILL|GRINDER|NAIL|WELD|HAMMER|LUMBER|MACHIN|"
    r"FORK ?LIFT|CONVEYOR|WRENCH|SCREWDRIVER|PLIERS|CROWBAR|\bAX|CHISEL|COMPRESSOR|"
    r"GENERATOR|WORKSHOP|POWER TOOL|HAND TOOL|BOX CUTTER|UTILITY KNI|GLASS|DOORS|WINDOWS|"
    r"PIPE|WIRE|METAL|BUCKET|CARTS",
    re.IGNORECASE,
)


def label_in_text(narrative: str, label: str) -> bool:
    """EDA heuristic (milestone1_eda.ipynb): a 4+ letter word of the NEISS label appears."""
    label_words = set(re.findall(r"[A-Z]{4,}", re.sub(r"\(.*?\)", "", str(label).upper()))) - _LABEL_SKIP
    text_words = {t.rstrip("S") for t in re.findall(r"[A-Z]{4,}", str(narrative).upper())}
    return any(w.rstrip("S") in text_words for w in label_words)


def add_slices(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["label_in_text"] = [label_in_text(t, l) for t, l in zip(out["narrative"], out["product_label"])]
    out["workplace_like"] = out["product_label"].fillna("").str.contains(WORKPLACE_PATTERN)
    out["multi_product"] = out["gold_set"].map(len) > 1
    return out


SLICES = {
    "all": lambda d: pd.Series(True, index=d.index),
    # The EDA "vague" flag: N.E.C. / NOT SPECIFIED labels, mostly default codes for
    # common products (beds, chairs, doors, knives), so not the same as uninformative.
    "label without catch-all wording": lambda d: ~d["vague_label"].astype(bool),
    "catch-all label (EDA vague flag)": lambda d: d["vague_label"].astype(bool),
    "label word in narrative": lambda d: d["label_in_text"],
    "label word not in narrative": lambda d: ~d["label_in_text"],
    "single product": lambda d: ~d["multi_product"],
    "multiple products": lambda d: d["multi_product"],
    "redacted text": lambda d: d["redacted"].astype(bool),
    "workplace-like product": lambda d: d["workplace_like"],
}


def slice_table(scored: pd.DataFrame, systems) -> pd.DataFrame:
    rows = []
    for slice_name, pick in SLICES.items():
        part = scored[pick(scored)]
        if part.empty:
            continue
        for s in systems:
            m = micro(part[f"{s}_correct"].values, part[f"{s}_answered"].values)
            rows.append({"slice": slice_name, "system": s, "n": m["n"],
                         "micro_precision": m["precision"], "micro_recall": m["recall"],
                         "micro_f1": m["f1"], "set_match": float(part[f"{s}_set_match"].mean())})
    return pd.DataFrame(rows)


def error_category(row, name: str = FINAL) -> str:
    if row[f"{name}_correct"]:
        return "correct"
    code = row[f"{name}_code"]
    if code == ABSTAIN:
        return "abstained (OTHER)"
    if code == UNMAPPED:
        return "span has no lexicon code"
    if row[f"{name}_set_match"]:
        return "matches a secondary NEISS product"
    entity_type = str(row[f"{name}_entity_type"])
    if entity_type.startswith("rule:"):
        return "floor/keyword rule wrong"
    if entity_type == "classifier":
        return "different product"
    return "model span, different product"


def error_table(scored: pd.DataFrame, name: str) -> pd.DataFrame:
    cats = scored[f"{name}_error_category"]
    out = cats.value_counts().rename_axis("category").reset_index(name="cases")
    out["share_of_errors"] = np.where(out["category"] == "correct", np.nan,
                                      out["cases"] / (cats != "correct").sum())
    return out


def rule_table(scored: pd.DataFrame, name: str = FINAL) -> pd.DataFrame:
    """Held-out check of the fallback rules (Task #2 measured these in-sample)."""
    rules = scored[scored[f"{name}_entity_type"].fillna("").astype(str).str.startswith("rule:")]
    if rules.empty:
        return pd.DataFrame(columns=["rule", "answer", "fired", "match_product_1", "match_any"])
    return (rules.groupby([f"{name}_entity_type", f"{name}_span"])
            .agg(fired=(f"{name}_correct", "size"), match_product_1=(f"{name}_correct", "sum"),
                 match_any=(f"{name}_set_match", "sum"))
            .reset_index().rename(columns={f"{name}_entity_type": "rule", f"{name}_span": "answer"})
            .sort_values("fired", ascending=False))


def confusions(scored: pd.DataFrame, names: Dict[int, str], name: str = FINAL, top: int = 15) -> pd.DataFrame:
    wrong = scored[~scored[f"{name}_correct"] & scored[f"{name}_answered"]
                   & (scored[f"{name}_code"] != UNMAPPED)]
    pairs = wrong.groupby(["product_code", f"{name}_code"]).size().rename("cases").reset_index()
    # Ties broken by code, with a stable sort, so the table is the same on every machine.
    pairs = pairs.sort_values(["cases", "product_code", f"{name}_code"], ascending=[False, True, True],
                              kind="stable").head(top)
    pairs["gold"] = pairs["product_code"].map(lambda c: names.get(int(c), str(c)))
    pairs["predicted"] = pairs[f"{name}_code"].map(lambda c: names.get(int(c), str(c)))
    return pairs[["gold", "predicted", "cases"]]


def win_loss(scored: pd.DataFrame, first: str, second: str) -> dict:
    a, b = scored[f"{first}_correct"], scored[f"{second}_correct"]
    return {"comparison": f"{first} vs {second}", "both right": int((a & b).sum()),
            "only first right": int((a & ~b).sum()), "only second right": int((~a & b).sum()),
            "both wrong": int((~a & ~b).sum())}


def hospital_table(scored: pd.DataFrame, systems) -> pd.DataFrame:
    """Micro F1 per test-fold hospital: how much house style moves the score."""
    rows = []
    for psu, part in scored.groupby("psu"):
        row = {"psu": int(psu), "n": len(part)}
        for s in systems:
            row[s] = micro(part[f"{s}_correct"].values, part[f"{s}_answered"].values)["f1"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# ---------------------------------------------------------------- dev reproduction and hand labels


def dev_check(dev: pd.DataFrame, preds: Dict[str, pd.DataFrame], mapper: CodeMapper) -> pd.DataFrame:
    """Re-score each version on the 25 development cases and compare with the earlier reports."""
    rows = []
    for name in PIPELINES:
        p = preds[name].rename(columns={"span": "extracted_object"})[["case_number", "extracted_object"]]
        scored = evaluate.score(dev.merge(p, on="case_number", how="left"))
        m = evaluate.metrics(scored, "strict")
        rows.append({"system": name, "TP": m["TP"], "FP": m["FP"], "FN": m["FN"],
                     "f1": m["f1"], "reported": REPORTED_DEV_COUNTS.get(name),
                     "reproduced": (m["TP"], m["FP"], m["FN"]) == REPORTED_DEV_COUNTS.get(name)})
    gold_codes = pd.to_numeric(dev["object_code"], errors="coerce")
    # "floor (concrete)" -> "floor", "stairs/steps" -> "stairs": the name before any qualifier.
    mapped = [mapper.map_span(re.split(r"[/,]", re.sub(r"\(.*?\)", " ", g))[0]) for g in dev["gold_object"]]
    rows.append({"system": "lexicon: gold name -> labeler's code", "TP": int((gold_codes.values == mapped).sum()),
                 "FP": "", "FN": "", "f1": np.nan,
                 "reported": f"of {len(dev)} cases", "reproduced": ""})
    return pd.DataFrame(rows)


def load_hand_labels(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = evaluate.load_ground_truth(path)
    df = df[df["gold_object"].fillna("").str.len() > 0]
    return df if len(df) else None


def hand_label_metrics(hand: pd.DataFrame, preds: Dict[str, pd.DataFrame], mapper: CodeMapper) -> pd.DataFrame:
    rows = []
    gold_codes = pd.to_numeric(hand["object_code"], errors="coerce").fillna(0).astype(int).values
    for name, p in preds.items():
        p = p.rename(columns={"span": "extracted_object"})[["case_number", "extracted_object", "code"]]
        scored = evaluate.score(hand.merge(p, on="case_number", how="left"))
        for rule in ("strict", "lenient"):
            m = evaluate.metrics(scored, rule)
            rows.append({"system": name, "scoring": f"span {rule}", "n": m["n"], "precision": m["precision"],
                         "recall": m["recall"], "f1": m["f1"], "exact_match": m["TP"] / m["n"]})
        codes = scored["code"].fillna(UNMAPPED).astype(int).values
        mi = micro(codes == gold_codes, codes != ABSTAIN)
        ma = macro(gold_codes, codes)
        rows.append({"system": name, "scoring": "code micro", "n": mi["n"], "precision": mi["precision"],
                     "recall": mi["recall"], "f1": mi["f1"], "exact_match": mi["TP"] / mi["n"]})
        rows.append({"system": name, "scoring": "code macro", "n": mi["n"], "precision": ma["precision"],
                     "recall": ma["recall"], "f1": ma["f1"], "exact_match": np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    # astype(object) keeps ints as ints; iterrows would upcast a mixed row to float.
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in df.astype(object).iterrows()]
    return "\n".join(lines)


def write_summary(path: Path, ctx: dict) -> None:
    m = ctx["metrics"]
    headline = m[["system", "answered", "micro_precision", "micro_recall", "micro_f1",
                  "macro_precision", "macro_recall", "macro_f1", f"macro_f1_support{MACRO_MIN_SUPPORT}",
                  "exact_match_product_1", "exact_match_product_set", "set_match"]]
    ci = ctx["bootstrap"]
    lines = [
        "# Final held-out evaluation: generated tables",
        "",
        "Generated by `python -m src.final_eval`. The write-up is in `reports/final_evaluation.md`.",
        "",
        f"- Test set: `{TEST_SET_PATH}`, {ctx['n_test']:,} narratives from the "
        f"{ctx['n_test_psu']} test-fold hospitals (seed {TEST_SEED}).",
        f"- Lexicon: {ctx['lexicon_terms']:,} terms from {ctx['n_train']:,} training-fold narratives "
        f"(min support {LEXICON_MIN_SUPPORT}).",
        f"- Majority baseline predicts code {ctx['majority_code']} ({ctx['majority_label']}).",
        "",
        "## Development-set reproduction (25 cases, strict)",
        "",
        md_table(ctx["dev"]),
        "",
        "## Metrics against NEISS codes",
        "",
        md_table(headline),
        "",
        f"Macro averages cover {int(m['macro_classes'].iloc[0])} gold classes; the "
        f"`support{MACRO_MIN_SUPPORT}` column covers the "
        f"{int(m[f'macro_classes_support{MACRO_MIN_SUPPORT}'].iloc[0])} classes with at least "
        f"{MACRO_MIN_SUPPORT} test cases.",
        "",
        f"## Bootstrap 95% intervals ({BOOTSTRAP_ROUNDS} resamples, estimate ± 1.96 SD)",
        "",
        md_table(ci),
        "",
        "## Slices (micro, against Product_1)",
        "",
        md_table(ctx["slices"].pivot(index="slice", columns="system", values="micro_f1")
                 .reindex(list(SLICES)).dropna(how="all")
                 .join(ctx["slices"][ctx["slices"]["system"] == FINAL].set_index("slice")["n"])
                 .reset_index()[["slice", "n", *ctx["systems"]]]),
        "",
        "## Win/loss against Product_1",
        "",
        md_table(pd.DataFrame(ctx["win_loss"])),
        "",
        "## Micro F1 by test-fold hospital (PSU)",
        "",
        md_table(ctx["hospitals"]),
        "",
    ]
    for name, errors in ctx["errors"].items():
        lines += [
            f"## {name} error categories",
            "",
            md_table(errors),
            "",
            f"## {name} most common confusions (gold -> predicted)",
            "",
            md_table(ctx["confusions"][name]),
            "",
        ]
    lines += [
        f"## {OBJECT_MODEL} fallback rules on held-out data",
        "",
        md_table(ctx["rules"]),
        "",
        f"## {FINAL} per-class F1, classes with at least {MACRO_MIN_SUPPORT} test cases "
        f"({ERROR_SYSTEMS[1]} F1 for comparison)",
        "",
        md_table(ctx["per_class"]),
        "",
        "## Hand-labeled test subset",
        "",
    ]
    if ctx["hand"] is None:
        lines += ["Not used: no hand labels were collected for the test set. To add them, run "
                  "`python -m src.final_eval --make-label-sheet`, label "
                  f"`{HAND_LABEL_PATH}` with `data/eval/Labeling_Instructions.md`, and rerun.", ""]
    else:
        lines += [md_table(ctx["hand"]), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


SYSTEM_NAMES = {"v1": "v1 (Task #7)", "v2": "v2 (Task #9)", "v3": "v3 (Task #2)",
                "v4": "v4 GLiNER + rules", "classifier": "v4 classifier (final)",
                "keyword": "Keyword lookup", "majority": "Majority class"}


def plot_comparison(metrics_df: pd.DataFrame, ci: pd.DataFrame, path: Path) -> None:
    """Two panels on one 0-1 scale: micro F1 and macro F1 per system, with 95% bootstrap bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid, series = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    order = [s for s in ("majority", "keyword", "v1", "v2", "v3", "v4", "classifier")
             if s in set(metrics_df["system"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 1.2 + 0.5 * len(order)), sharey=True, facecolor=surface)
    for ax, metric, title in zip(axes, ("micro_f1", "macro_f1"), ("Micro F1", "Macro F1")):
        ax.set_facecolor(surface)
        for y, s in enumerate(order):
            value = float(metrics_df.loc[metrics_df["system"] == s, metric].iloc[0])
            row = ci[(ci["system"] == s) & (ci["metric"] == metric)]
            if not row.empty:
                lo, hi = row["low"].iloc[0], row["high"].iloc[0]
                ax.plot([lo, hi], [y, y], color=series, lw=2, solid_capstyle="round")
                ax.plot([lo, lo], [y - 0.14, y + 0.14], color=series, lw=2)
                ax.plot([hi, hi], [y - 0.14, y + 0.14], color=series, lw=2)
            ax.plot(value, y, "o", ms=7, color=series, mec=surface, mew=1.5, zorder=3)
            ax.text(value, y + 0.28, f"{value:.2f}", ha="center", va="bottom", fontsize=9, color=ink)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, len(order) - 0.3)
        ax.set_yticks(range(len(order)), [SYSTEM_NAMES.get(s, s) for s in order], color=ink)
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.tick_params(colors=muted, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    fig.suptitle("Held-out test fold, scored against NEISS Product_1 (bars: 95% interval, ±1.96 bootstrap SD)",
                 x=0.01, ha="left", fontsize=10, color=muted)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


# ---------------------------------------------------------------- main


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final held-out evaluation")
    p.add_argument("--clean", type=Path, default=extract.DEFAULT_INPUT)
    p.add_argument("--fmt", type=Path, default=FMT_PATH)
    p.add_argument("--split", type=Path, default=SPLIT_PATH)
    p.add_argument("--dev-gold", type=Path, default=evaluate.DEFAULT_GROUND_TRUTH)
    p.add_argument("--test-set", type=Path, default=TEST_SET_PATH)
    p.add_argument("--hand-labels", type=Path, default=HAND_LABEL_PATH)
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--details", type=Path, default=DETAILS_PATH)
    p.add_argument("--metrics", type=Path, default=METRICS_PATH)
    p.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    p.add_argument("--error-sample-dir", type=Path, default=ERROR_SAMPLE_DIR,
                   help="folder for final_eval_error_sample_<system>.csv (existing files are kept)")
    p.add_argument("--figure", type=Path, default=FIGURE_PATH)
    p.add_argument("--make-label-sheet", action="store_true",
                   help=f"write a blank {HAND_N}-case hand-labeling sheet if it does not exist")
    p.add_argument("--n", type=int, default=TEST_N)
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP_ROUNDS)
    p.add_argument("--rebuild-test-set", action="store_true",
                   help="resample the test set even if the file exists (the hand-label sheet is never overwritten)")
    return p.parse_args(argv)


def main(argv=None, model_loader: Callable = extract.load_model) -> dict:
    args = parse_args(argv)
    clean = extract.load_clean(args.clean)
    clean["case_number"] = clean["case_number"].astype(str)
    folds = load_folds(args.split)
    names = load_product_names(args.fmt)
    dev = evaluate.load_ground_truth(args.dev_gold)

    if args.rebuild_test_set or not args.test_set.exists():
        test = build_test_set(clean, folds, development_case_numbers(clean, dev), n=args.n)
        args.test_set.parent.mkdir(parents=True, exist_ok=True)
        test.to_csv(args.test_set, index=False)
        print(f"Wrote {len(test):,} test cases to {args.test_set}")
    test = pd.read_csv(args.test_set, dtype={"case_number": str})
    leaked = set(test["case_number"]) & set(dev["case_number"])
    if leaked:
        raise SystemExit(f"Development cases found in the test set: {sorted(leaked)}")
    if args.make_label_sheet and not args.hand_labels.exists():
        hand_label_template(test).to_csv(args.hand_labels, index=False)
        print(f"Wrote a blank {HAND_N}-case labeling sheet to {args.hand_labels}")

    # Every training-fold row, including the EDA's "vague" labels: that flag mostly
    # marks the default code for common products (BEDS ... NOT SPECIFIED, KNIVES N.E.C.),
    # and dropping them would leave "bed" or "knife" with no code to map to.
    fold = clean["psu"].astype(int).map(folds)
    train = clean[fold == "train"]
    print(f"Building the span -> code lexicon from {len(train):,} training-fold narratives ...")
    mapper = CodeMapper.fit(train["event_text"], train["product_code"])
    majority_code = int(train["product_code"].mode().iloc[0])
    majority_label = names.get(majority_code, str(majority_code))
    majority_span = evaluate._words(majority_label)[0].upper()

    try:
        from src import classify
    except ImportError:
        import classify
    print("Training the v4 classifier on the training fold ...")
    classifier = classify.ProductClassifier.fit(train["event_text"], train["product_code"])

    model = LazyModel(model_loader)
    both = pd.concat([test[["case_number", "narrative"]], dev[["case_number", "narrative"]]])
    preds_all = predict_all(both, mapper, majority_code, majority_span, model, args.cache_dir,
                            classifier=classifier)
    test_ids, dev_ids = set(test["case_number"]), set(dev["case_number"])
    preds = {k: v[v["case_number"].isin(test_ids)] for k, v in preds_all.items()}
    dev_preds = {k: v[v["case_number"].isin(dev_ids)] for k, v in preds_all.items()}

    scored = add_slices(score_codes(test, preds))
    systems = list(preds)
    gold = scored["product_code"].astype(int)
    frequent = gold.value_counts()
    frequent = np.array(frequent[frequent >= MACRO_MIN_SUPPORT].index)
    metrics_df = pd.DataFrame([system_metrics(scored, s, frequent) for s in systems])

    error_systems = [s for s in ERROR_SYSTEMS if s in systems]
    for s in error_systems:
        scored[f"{s}_error_category"] = scored.apply(error_category, axis=1, name=s)
    pc = per_class(gold.values, scored[f"{FINAL}_code"].values)
    if len(ERROR_SYSTEMS) > 1 and ERROR_SYSTEMS[1] in systems:
        other = ERROR_SYSTEMS[1]
        pc[f"{other}_f1"] = per_class(gold.values, scored[f"{other}_code"].values)["f1"].values
    pc = pc[pc["support"] >= MACRO_MIN_SUPPORT].sort_values("f1", ascending=False)
    pc.insert(1, "product", pc["code"].map(lambda c: names.get(int(c), str(c))))

    hand = load_hand_labels(args.hand_labels)
    ctx = {
        "systems": systems, "metrics": metrics_df,
        "bootstrap": bootstrap(scored, systems, rounds=args.bootstrap),
        "slices": slice_table(scored, systems),
        "win_loss": [win_loss(scored, a, b) for a, b in comparisons(systems)],
        "hospitals": hospital_table(scored, systems),
        "errors": {s: error_table(scored, s) for s in error_systems},
        "confusions": {s: confusions(scored, names, name=s) for s in error_systems},
        "rules": rule_table(scored, name=OBJECT_MODEL),
        "per_class": pc, "dev": dev_check(dev, dev_preds, mapper),
        "hand": hand_label_metrics(hand, preds, mapper) if hand is not None else None,
        "n_test": len(test), "n_test_psu": test["psu"].nunique(), "n_train": len(train),
        "lexicon_terms": len(mapper.table), "majority_code": majority_code,
        "majority_label": majority_label,
    }

    for s in systems:
        scored[f"{s}_label"] = scored[f"{s}_code"].map(
            lambda c: {ABSTAIN: "OTHER", UNMAPPED: "UNMAPPED"}.get(int(c), names.get(int(c), str(c))))
    detail_cols = (["case_number", "psu", "narrative", "product_code", "product_label", "product_2_code",
                    "product_3_code", "vague_label", "redacted", "label_in_text", "workplace_like",
                    "multi_product"]
                   + [f"{s}_error_category" for s in error_systems]
                   + [f"{s}_{f}" for s in systems for f in ("span", "entity_type", "code", "label",
                                                            "correct", "set_match")])
    args.details.parent.mkdir(parents=True, exist_ok=True)
    scored[detail_cols].to_csv(args.details, index=False)
    metrics_df.to_csv(args.metrics, index=False)
    for s in error_systems:
        path = args.error_sample_dir / f"final_eval_error_sample_{s}.csv"
        if path.exists():  # may hold a hand review
            continue
        errs = scored[scored[f"{s}_error_category"] != "correct"]
        sample = errs.sample(n=min(ERROR_SAMPLE_N, len(errs)), random_state=TEST_SEED)
        # The classifier names no object, so show the v4 object next to its code.
        spans = [f"{OBJECT_MODEL}_span", f"{OBJECT_MODEL}_label"] if s == FINAL else [f"{s}_span"]
        sample = sample[["case_number", "narrative", "product_label", "product_2_code", "product_3_code",
                         *spans, f"{s}_label", f"{s}_error_category"]].copy()
        sample = sample.rename(columns={f"{s}_error_category": "error_category"})
        sample["review_category"] = ""
        sample["review_note"] = ""
        args.error_sample_dir.mkdir(parents=True, exist_ok=True)
        sample.to_csv(path, index=False)
    write_summary(args.summary, ctx)
    if args.figure is not None:
        plot_comparison(metrics_df, ctx["bootstrap"], args.figure)

    show = metrics_df[["system", "micro_precision", "micro_recall", "micro_f1", "macro_f1",
                       "exact_match_product_1", "set_match"]]
    print()
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {args.details}, {args.metrics} and {args.summary}")
    return ctx


if __name__ == "__main__":
    main()
