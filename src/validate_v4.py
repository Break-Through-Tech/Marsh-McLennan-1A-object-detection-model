"""Validation harness for v4 (Milestone 3).

Scores candidate pipelines on narratives from the 10 validation-fold hospitals
in reports/split_by_psu.csv. Nothing here reads the 3,000-case test set, so the
final v3 vs v4 comparison in src/final_eval.py stays held out.

Uses the same span -> code lexicon (built from the training fold) and the same
metrics as src/final_eval.py.

Run from the repo root (needs data/processed/neiss2025_clean.csv):

    python -m src.validate_v4                 # 5,000 validation cases
    python -m src.validate_v4 --n 2000

GLiNER output is cached in data/processed/val_cache/, so a rerun only rescores.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

try:  # python -m src.validate_v4
    from src import evaluate, extract, final_eval
except ImportError:  # python src/validate_v4.py
    import evaluate
    import extract
    import final_eval

VAL_N = 5000
VAL_SEED = 404
CACHE_DIR = Path("data/processed/val_cache")
ABSTAIN, UNMAPPED = final_eval.ABSTAIN, final_eval.UNMAPPED


def build_val_set(clean: pd.DataFrame, folds: Dict[int, str], exclude, n: int = VAL_N,
                  seed: int = VAL_SEED) -> pd.DataFrame:
    """Random validation-fold sample, minus development cases and repeated narratives."""
    fold = clean["psu"].astype(int).map(folds)
    pool = clean[(fold == "val") & ~clean["case_number"].astype(str).isin(set(map(str, exclude)))]
    pool = pool.drop_duplicates("narrative")
    out = pool.sample(n=min(n, len(pool)), random_state=seed).copy()
    out["case_number"] = out["case_number"].astype(str)
    return out.reset_index(drop=True)


def lookup_predict(mapper: final_eval.CodeMapper, text: str):
    """Like CodeMapper.keyword_predict, plus the purity of the winning term (its confidence)."""
    best = None
    for term in final_eval.terms(text):
        if term in mapper:
            row = mapper.table.loc[term]
            key = (float(row["purity"]), int(row["support"]), term)
            if best is None or key > best:
                best = key
    if best is None:
        return extract.FALLBACK, ABSTAIN, 0.0
    return best[2].upper(), int(mapper._code[best[2]]), best[0]


def pipeline_preds(df: pd.DataFrame, config: extract.ExtractionConfig, cache: dict,
                   mapper: final_eval.CodeMapper) -> pd.DataFrame:
    """case_number, span, entity_type, confidence, code for one GLiNER pipeline."""
    p = final_eval.run_pipeline(df, config, cache).rename(
        columns={"extracted_object": "span", "extracted_entity_type": "entity_type",
                 "extraction_confidence": "confidence"})
    abstain = p["span"].fillna(extract.FALLBACK).str.upper() == extract.FALLBACK
    p["code"] = [ABSTAIN if a else mapper.map_span(s) for s, a in zip(p["span"], abstain)]
    return p.reset_index(drop=True)


def lookup_preds(df: pd.DataFrame, mapper: final_eval.CodeMapper) -> pd.DataFrame:
    events = extract.prepare_text(df["narrative"], extract.ExtractionConfig())["event_text"]
    rows = [lookup_predict(mapper, t) for t in events]
    return pd.DataFrame({
        "case_number": df["case_number"].values, "span": [r[0] for r in rows],
        "entity_type": "lookup", "confidence": [r[2] for r in rows], "code": [r[1] for r in rows]})


def combine(primary: pd.DataFrame, secondary: pd.DataFrame,
            use_primary: Callable[[pd.Series], bool]) -> pd.DataFrame:
    """Row by row: take primary where use_primary(row) is true, else secondary."""
    a = primary.set_index("case_number")
    b = secondary.set_index("case_number").reindex(a.index)
    mask = np.array([bool(use_primary(r)) for _, r in a.iterrows()])
    out = b.copy()
    out.loc[mask] = a.loc[mask]
    return out.reset_index()


def model_answered(p: pd.DataFrame) -> pd.Series:
    """GLiNER gave a span (not a rule, not OTHER) that the lexicon could map."""
    rule = p["entity_type"].fillna("").astype(str).str.startswith("rule:")
    return (p["code"] >= 0) & ~rule


def score(val: pd.DataFrame, preds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    scored = final_eval.score_codes(val, preds)
    gold = scored["product_code"].astype(int)
    counts = gold.value_counts()
    frequent = np.array(counts[counts >= final_eval.MACRO_MIN_SUPPORT].index)
    rows = [final_eval.system_metrics(scored, s, frequent) for s in preds]
    cols = ["system", "answered", "micro_precision", "micro_recall", "micro_f1", "macro_f1",
            "set_match"]
    return pd.DataFrame(rows)[cols]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score v4 candidates on validation hospitals")
    p.add_argument("--clean", type=Path, default=extract.DEFAULT_INPUT)
    p.add_argument("--n", type=int, default=VAL_N)
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    return p.parse_args(argv)


def setup(argv=None, model_loader=extract.load_model):
    """Load data, build the validation set and the lexicon. Returns a context dict."""
    args = parse_args(argv)
    clean = extract.load_clean(args.clean)
    clean["case_number"] = clean["case_number"].astype(str)
    folds = final_eval.load_folds()
    dev = evaluate.load_ground_truth(final_eval.DEFAULT_DEV_GOLD) if hasattr(final_eval, "DEFAULT_DEV_GOLD") \
        else evaluate.load_ground_truth(evaluate.DEFAULT_GROUND_TRUTH)
    val = build_val_set(clean, folds, final_eval.development_case_numbers(clean, dev), n=args.n)
    test = pd.read_csv(final_eval.TEST_SET_PATH, dtype={"case_number": str})
    assert not set(val["case_number"]) & set(test["case_number"]), "validation overlaps the test set"
    train = clean[clean["psu"].astype(int).map(folds) == "train"]
    mapper = final_eval.CodeMapper.fit(train["event_text"], train["product_code"])
    return {"args": args, "val": val, "train": train, "mapper": mapper,
            "model": final_eval.LazyModel(model_loader)}


def gliner_cache(ctx, config: extract.ExtractionConfig) -> dict:
    return final_eval.entity_cache(ctx["val"]["narrative"], config, ctx["model"], ctx["args"].cache_dir)


def main(argv=None) -> pd.DataFrame:
    try:
        from src import classify
    except ImportError:
        import classify
    ctx = setup(argv)
    val, mapper = ctx["val"], ctx["mapper"]
    cache = gliner_cache(ctx, final_eval.PIPELINES["v4"])  # v3 and v4 share labels and text
    v3 = pipeline_preds(val, final_eval.PIPELINES["v3"], cache, mapper)
    v4 = pipeline_preds(val, final_eval.PIPELINES["v4"], cache, mapper)
    lookup = lookup_preds(val, mapper)
    print(f"Training the classifier on {len(ctx['train']):,} training-fold narratives ...")
    clf = classify.ProductClassifier.fit(ctx["train"]["event_text"], ctx["train"]["product_code"])
    codes, margins = clf.predict(val["event_text"])
    classifier = pd.DataFrame({"case_number": val["case_number"].values, "span": "",
                               "entity_type": "classifier", "confidence": margins, "code": codes})
    confident = lookup.copy()
    gl = v4.copy()
    gl.loc[~model_answered(v4).values, "code"] = ABSTAIN
    hybrid = combine(combine(confident, gl, lambda r: r["confidence"] >= 0.7), lookup, lambda r: r["code"] >= 0)
    preds = {"v3 (Task #2)": v3, "v4 GLiNER + rules": v4, "lookup": lookup,
             "lookup >= 0.7, else GLiNER, else lookup": hybrid, "classifier (v4)": classifier}
    table = score(val, preds)
    print(f"\n{len(val):,} validation-fold narratives\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return table


if __name__ == "__main__":
    main()
