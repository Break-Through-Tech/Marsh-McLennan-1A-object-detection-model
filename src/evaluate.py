"""Evaluation of the v1 extraction pipeline (Milestone 2, Task #7).

Runs the v1 pipeline on the hand-labeled ground truth sample (Task #6) and
scores the predictions with precision, recall and F1. Writes a per-case CSV
that Task #8 (error analysis) can start from.

Run from the repo root:

    python -m src.evaluate                                  # runs GLiNER, then scores
    python -m src.evaluate --predictions my_preds.csv       # score an existing CSV only

How a prediction is scored
    The model returns a text span ("STEPS"); the labeler wrote a short name
    ("stairs/steps"). Both are lowercased, punctuation is removed and plurals
    are trimmed, then two match rules are applied:
      strict   the span equals one of the gold names
               ("stairs/steps" gives the names "stairs" and "steps")
      lenient  the span and the gold label share at least one content word
               ("tile floor" vs "floor (tile)")
    "(activity)" in a gold label is ignored, so "basketball" matches
    "basketball (activity)".

How the metrics are defined (one gold object per narrative)
    The model either predicts an object or abstains with OTHER.
      TP  predicted an object and it matches the gold object
      FP  predicted an object and it does not match
      FN  the gold object was not found (wrong prediction or OTHER)
    precision = TP / (TP + FP)   of the objects the model returned, how many were right
    recall    = TP / (TP + FN)   of the labeled objects, how many the model found
    F1        = harmonic mean of precision and recall
    A wrong prediction counts as both an FP and an FN, which is the usual
    convention for single-answer extraction.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd

try:  # python -m src.evaluate
    from src import extract
except ImportError:  # python src/evaluate.py
    import extract

DEFAULT_GROUND_TRUTH = Path("data/eval/ground_truth_sample_1.csv")
DEFAULT_DETAILS = Path("reports/task7_eval_details.csv")
DEFAULT_SUMMARY = Path("reports/task7_eval_results.md")

# Words in a gold label that describe the label rather than name the object.
IGNORE_WORDS = frozenset({"activity", "the", "a", "an", "of", "s"})


# ---------------------------------------------------------------- loading

def load_ground_truth(path: Path) -> pd.DataFrame:
    """Read the Task #6 CSV. It is padded with spaces, so strip every cell."""
    df = pd.read_csv(path, skipinitialspace=True, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].str.strip()
    df = df.dropna(subset=["case_number", "gold_object"])
    df["case_number"] = df["case_number"].astype(str)
    return df.reset_index(drop=True)


def load_predictions(path: Path) -> pd.DataFrame:
    """Read a predictions CSV with case_number and extracted_object columns."""
    df = pd.read_csv(path, dtype={"case_number": str})
    missing = {"case_number", "extracted_object"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    df["case_number"] = df["case_number"].astype(str).str.strip()
    return df


# ---------------------------------------------------------------- matching

def _singular(word: str) -> str:
    """Trim a plural ending: stairs -> stair, knives -> knife, boxes -> box."""
    if len(word) > 4 and word.endswith("ves"):
        return word[:-3] + "fe"
    if len(word) > 4 and word.endswith(("ses", "xes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text: str) -> list:
    text = re.sub(r"[^a-z0-9 ]+", " ", str(text).lower())
    return [_singular(w) for w in text.split() if w and w not in IGNORE_WORDS]


def gold_names(gold: str) -> list:
    """'stairs/steps' -> ['stair', 'step']; 'floor (tile)' -> ['floor', 'tile', 'floor tile']."""
    names = set()
    no_parens = re.sub(r"\(.*?\)", " ", gold)
    for part in re.split(r"[/,]", no_parens):
        words = _words(part)
        if words:
            names.add(" ".join(words))
    inside = re.findall(r"\((.*?)\)", gold)
    for part in inside:
        words = _words(part)
        if words:
            names.add(" ".join(words))
    return sorted(names)


def strict_match(pred: str, gold: str) -> bool:
    return " ".join(_words(pred)) in gold_names(gold)


def lenient_match(pred: str, gold: str) -> bool:
    return bool(set(_words(pred)) & set(_words(gold)))


# ---------------------------------------------------------------- scoring

def score(merged: pd.DataFrame, fallback: str = extract.FALLBACK) -> pd.DataFrame:
    """Add abstained / strict_correct / lenient_correct columns."""
    out = merged.copy()
    pred = out["extracted_object"].fillna(fallback).astype(str).str.strip()
    out["abstained"] = pred.str.upper() == fallback.upper()
    out["strict_correct"] = [
        (not a) and strict_match(p, g) for p, g, a in zip(pred, out["gold_object"], out["abstained"])
    ]
    out["lenient_correct"] = [
        (not a) and lenient_match(p, g) for p, g, a in zip(pred, out["gold_object"], out["abstained"])
    ]
    return out


def metrics(scored: pd.DataFrame, rule: str) -> dict:
    correct = scored[f"{rule}_correct"]
    answered = ~scored["abstained"]
    tp = int((correct & answered).sum())
    fp = int((answered & ~correct).sum())
    fn = int(len(scored) - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(scored), "answered": int(answered.sum()), "TP": tp, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def metrics_table(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in ("strict", "lenient"):
        rows.append({"subset": "all", "rule": rule, **metrics(scored, rule)})
    if "mention_type" in scored.columns:
        for mtype, group in scored.groupby("mention_type"):
            for rule in ("strict", "lenient"):
                rows.append({"subset": f"mention_type={mtype}", "rule": rule, **metrics(group, rule)})
    return pd.DataFrame(rows)


def result_label(row) -> str:
    if row["abstained"]:
        return "no prediction (OTHER)"
    if row["strict_correct"]:
        return "correct"
    if row["lenient_correct"]:
        return "partial match"
    return "wrong object"


# ---------------------------------------------------------------- report

def _fmt_metrics(table: pd.DataFrame) -> str:
    t = table.copy()
    for col in ("precision", "recall", "f1"):
        t[col] = t[col].map(lambda x: f"{x:.2f}")
    cols = ["subset", "rule", "n", "answered", "TP", "FP", "FN", "precision", "recall", "f1"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in t.iterrows()]
    return "\n".join([header, sep] + body)


def write_summary(path: Path, table: pd.DataFrame, details: pd.DataFrame, config_note: str) -> None:
    misses = details[details["result"] != "correct"]
    lines = [
        "# Task #7: Evaluation of extraction pipeline v1",
        "",
        f"Ground truth: `{DEFAULT_GROUND_TRUTH}` ({len(details)} hand-labeled narratives, Task #6).",
        f"Pipeline settings: {config_note}",
        "",
        "## Metrics",
        "",
        _fmt_metrics(table),
        "",
        "Strict = the extracted span is the gold object name. "
        "Lenient = they share a content word (e.g. \"tile floor\" vs \"floor (tile)\"). "
        "See `src/evaluate.py` for the exact rules.",
        "",
        "## Results by case",
        "",
        details["result"].value_counts().rename_axis("result").reset_index(name="cases")
        .to_markdown(index=False) if _has_tabulate() else details["result"].value_counts().to_string(),
        "",
        f"## Cases to review for error analysis (Task #8): {len(misses)}",
        "",
    ]
    for _, r in misses.iterrows():
        lines.append(
            f"- `{r['case_number']}` predicted **{r['extracted_object']}**, "
            f"gold **{r['gold_object']}** ({r['mention_type']}): {r['result']}"
        )
    lines += ["", f"Full per-case output: `{DEFAULT_DETAILS}`", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------- main

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task #7: score extraction v1 against ground truth")
    p.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    p.add_argument("--predictions", type=Path, default=None,
                   help="score this CSV instead of running the model")
    p.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--threshold", type=float, default=extract.DEFAULT_THRESHOLD)
    p.add_argument("--text-field", choices=["event", "normalized"], default="event")
    return p.parse_args(argv)


def run_model(gold: pd.DataFrame, config: extract.ExtractionConfig) -> pd.DataFrame:
    model = extract.load_model(config)
    return extract.run(gold[["case_number", "narrative"]], model, config)


def main(argv=None, model_runner=run_model) -> pd.DataFrame:
    args = parse_args(argv)
    gold = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(gold)} ground truth labels from {args.ground_truth}")

    if args.predictions is not None:
        preds = load_predictions(args.predictions)
        config_note = f"scored existing file `{args.predictions}`"
    else:
        config = extract.ExtractionConfig(threshold=args.threshold, text_field=args.text_field)
        print(f"Running {config.model_name} on the ground truth narratives ...")
        preds = model_runner(gold, config)
        config_note = (f"model `{config.model_name}`, labels {list(config.labels)}, "
                       f"threshold {config.threshold}, text field `{config.text_field}`")

    pred_cols = [c for c in ("case_number", "extracted_object", "extracted_entity_type",
                             "extraction_confidence") if c in preds.columns]
    merged = gold.merge(preds[pred_cols], on="case_number", how="left")
    unmatched = merged["extracted_object"].isna().sum()
    if unmatched == len(merged):
        raise SystemExit("No case numbers in the predictions match the ground truth file.")
    if unmatched:
        print(f"Warning: {unmatched} ground truth cases had no prediction; counted as OTHER.")

    scored = score(merged)
    scored["result"] = scored.apply(result_label, axis=1)
    table = metrics_table(scored)

    keep = ["case_number", "narrative", "gold_object", "object_code", "mention_type",
            "extracted_object", "extracted_entity_type", "extraction_confidence",
            "strict_correct", "lenient_correct", "result", "notes"]
    details = scored[[c for c in keep if c in scored.columns]]
    args.details.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.details, index=False)
    write_summary(args.summary, table, details, config_note)

    print()
    print(table[table["subset"] == "all"][["rule", "TP", "FP", "FN", "precision", "recall", "f1"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nWrote {args.details} and {args.summary}")
    return table


if __name__ == "__main__":
    main()