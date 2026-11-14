"""Model tuning for the v2 pipeline (Milestone 3, Task #2).

Tries combinations of GLiNER labels (the "prompt"), input text, and score
threshold on the hand-labeled cases, then reports precision, recall and F1.

The model runs once per (label set, text variant) at a low threshold and its
entities are cached. Thresholds and the rule switches are applied afterwards
on the cached entities, so the whole grid takes about as long as a few runs.

Run from the repo root:

    python -m src.tune

Writes:
    reports/task2_tuning_grid.csv     one row per configuration tried
    reports/task2_eval_details.csv    per-case results for the best configuration
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Optional

import pandas as pd

try:  # python -m src.tune
    from src import evaluate, extract
except ImportError:  # python src/tune.py
    import evaluate
    import extract

DEFAULT_GRID_OUT = Path("reports/task2_tuning_grid.csv")
DEFAULT_DETAILS_OUT = Path("reports/task2_eval_details.csv")

BASE_LABELS = ("object", "tool", "furniture", "sports equipment", "structure")  # the Task #9 labels
LABEL_SETS = {
    "base": BASE_LABELS,
    "+sport": BASE_LABELS + ("sport",),
    "+sport+vehicle": BASE_LABELS + ("sport", "vehicle"),
    "+sport+vehicle+household": BASE_LABELS + ("sport", "vehicle", "household item"),
}
# (text_field, lowercase)
TEXT_VARIANTS = [("event", False), ("event", True), ("normalized", False), ("normalized", True)]
THRESHOLDS = (0.2, 0.3, 0.4, 0.5)
CLEAN_OPTIONS = {"off": None, "on": extract.default_clean_span}
FALLBACK_OPTIONS = {"floor": extract.floor_for_falls, "keyword+floor": extract.keyword_then_floor}
CACHE_THRESHOLD = 0.1  # the model keeps everything above this; thresholds filter up from here


class CachedModel:
    """Looks like a GLiNER model but answers from cached entities at a threshold."""

    def __init__(self, cache: dict, threshold: float):
        self.cache = cache
        self.threshold = threshold

    def predict_entities(self, text, labels, threshold=0.5):
        return [e for e in self.cache[text] if float(e["score"]) >= self.threshold]


def build_cache(model, gold: pd.DataFrame, labels: tuple, text_field: str, lowercase: bool) -> dict:
    """Run the real model once at the cache threshold. Returns {model text: [entities]}."""
    config = extract.ExtractionConfig(
        labels=labels, threshold=CACHE_THRESHOLD, text_field=text_field, lowercase=lowercase
    )
    texts = extract.prepare_text(gold["narrative"], config)["model_text"].tolist()
    nonblank = [t for t in dict.fromkeys(texts) if t and t.strip()]
    entity_lists = extract.predict_entities(model, nonblank, config)
    return dict(zip(nonblank, entity_lists))


def evaluate_config(gold: pd.DataFrame, cache: dict, config: extract.ExtractionConfig) -> pd.DataFrame:
    """Run the pipeline on cached entities and score it. Returns the scored cases."""
    preds = extract.run(gold[["case_number", "narrative"]], CachedModel(cache, config.threshold), config)
    cols = ["case_number", "extracted_object", "extracted_entity_type", "extraction_confidence"]
    scored = evaluate.score(gold.merge(preds[cols], on="case_number", how="left"))
    scored["result"] = scored.apply(evaluate.result_label, axis=1)
    return scored


def run_grid(model, gold: pd.DataFrame) -> tuple:
    """Try every configuration. Returns (grid DataFrame, {config key: scored cases})."""
    rows, scored_by_key = [], {}
    for label_name, labels in LABEL_SETS.items():
        for text_field, lowercase in TEXT_VARIANTS:
            cache = build_cache(model, gold, labels, text_field, lowercase)
            options = itertools.product(THRESHOLDS, CLEAN_OPTIONS.items(), FALLBACK_OPTIONS.items())
            for threshold, (clean_name, clean_fn), (fallback_name, fallback_fn) in options:
                config = extract.ExtractionConfig(
                    labels=labels, threshold=threshold, text_field=text_field,
                    lowercase=lowercase, clean_span=clean_fn, fallback_rule=fallback_fn,
                )
                scored = evaluate_config(gold, cache, config)
                strict = evaluate.metrics(scored, "strict")
                lenient = evaluate.metrics(scored, "lenient")
                key = (label_name, text_field, lowercase, threshold, clean_name, fallback_name)
                scored_by_key[key] = scored
                rows.append({
                    "labels": label_name, "text_field": text_field, "lowercase": lowercase,
                    "threshold": threshold, "clean_span": clean_name, "fallback": fallback_name,
                    "strict_TP": strict["TP"], "strict_FP": strict["FP"], "strict_FN": strict["FN"],
                    "strict_precision": strict["precision"], "strict_recall": strict["recall"],
                    "strict_f1": strict["f1"],
                    "lenient_TP": lenient["TP"], "lenient_FP": lenient["FP"], "lenient_FN": lenient["FN"],
                    "lenient_precision": lenient["precision"], "lenient_recall": lenient["recall"],
                    "lenient_f1": lenient["f1"],
                })
    return pd.DataFrame(rows), scored_by_key


def baseline_key() -> tuple:
    """The Task #9 configuration: base labels, event text, threshold 0.3, no span cleaning, floor rule only."""
    return ("base", "event", False, 0.3, "off", "floor")


def key_of(row) -> tuple:
    return (row["labels"], row["text_field"], bool(row["lowercase"]), float(row["threshold"]),
            row["clean_span"], row["fallback"])


def rank(grid: pd.DataFrame) -> pd.DataFrame:
    """Best first. Ties go to the simpler setup: fewer labels, threshold nearest 0.3."""
    out = grid.copy()
    out["n_labels"] = out["labels"].map(lambda k: len(LABEL_SETS[k]))
    out["threshold_gap"] = (out["threshold"] - 0.3).abs()
    out = out.sort_values(
        ["strict_f1", "lenient_f1", "n_labels", "threshold_gap"],
        ascending=[False, False, True, True], kind="stable",
    )
    return out.drop(columns=["n_labels", "threshold_gap"]).reset_index(drop=True)


def case_changes(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Cases whose strict result changed between two scored runs."""
    b = before.set_index("case_number")
    a = after.set_index("case_number")
    changed = []
    for case in a.index:
        if bool(b.loc[case, "strict_correct"]) != bool(a.loc[case, "strict_correct"]):
            changed.append({
                "case_number": case, "gold_object": a.loc[case, "gold_object"],
                "before": b.loc[case, "extracted_object"], "after": a.loc[case, "extracted_object"],
                "change": "fixed" if a.loc[case, "strict_correct"] else "broken",
            })
    return pd.DataFrame(changed, columns=["case_number", "gold_object", "before", "after", "change"])


def save_details(path: Path, scored: pd.DataFrame) -> None:
    keep = ["case_number", "narrative", "gold_object", "object_code", "mention_type",
            "extracted_object", "extracted_entity_type", "extraction_confidence",
            "strict_correct", "lenient_correct", "result", "notes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    scored[[c for c in keep if c in scored.columns]].to_csv(path, index=False)


def describe(row) -> str:
    return (f"labels={row['labels']}, text={row['text_field']}, lowercase={row['lowercase']}, "
            f"threshold={row['threshold']}, clean_span={row['clean_span']}, fallback={row['fallback']}")


def fmt(row) -> str:
    return (f"strict P {row['strict_precision']:.2f} R {row['strict_recall']:.2f} F1 {row['strict_f1']:.2f} "
            f"(TP {int(row['strict_TP'])} FP {int(row['strict_FP'])} FN {int(row['strict_FN'])}) | "
            f"lenient P {row['lenient_precision']:.2f} R {row['lenient_recall']:.2f} F1 {row['lenient_f1']:.2f}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task #2: tune labels, text input and threshold")
    p.add_argument("--ground-truth", type=Path, default=evaluate.DEFAULT_GROUND_TRUTH)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID_OUT)
    p.add_argument("--details-out", type=Path, default=DEFAULT_DETAILS_OUT)
    return p.parse_args(argv)


def main(argv=None, model=None) -> pd.DataFrame:
    args = parse_args(argv)
    gold = evaluate.load_ground_truth(args.ground_truth)
    print(f"Loaded {len(gold)} labeled cases from {args.ground_truth}")
    if model is None:
        model = extract.load_model(extract.ExtractionConfig())

    grid, scored_by_key = run_grid(model, gold)
    ranked = rank(grid)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.grid_out, index=False)

    base_row = grid[grid.apply(lambda r: key_of(r) == baseline_key(), axis=1)].iloc[0]
    best_row = ranked.iloc[0]
    best_key = key_of(best_row)
    save_details(args.details_out, scored_by_key[best_key])

    print(f"\nTried {len(grid)} configurations.\n")
    print("Task #9 setup (the baseline for this task):")
    print("  " + describe(base_row)); print("  " + fmt(base_row))
    print("\nBest configuration:")
    print("  " + describe(best_row)); print("  " + fmt(best_row))
    near = int((ranked["strict_f1"] >= best_row["strict_f1"] - 0.02).sum())
    print(f"\nConfigurations within 0.02 strict F1 of the best: {near} of {len(ranked)}")

    print("\nTop 10:")
    show = ranked.head(10)[["labels", "text_field", "lowercase", "threshold", "clean_span", "fallback",
                            "strict_precision", "strict_recall", "strict_f1", "lenient_f1"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    changes = case_changes(scored_by_key[baseline_key()], scored_by_key[best_key])
    print("\nCases that changed (strict), baseline -> best:")
    print(changes.to_string(index=False) if len(changes) else "  none")
    print(f"\nWrote {args.grid_out} and {args.details_out}")
    return ranked


if __name__ == "__main__":
    main()
