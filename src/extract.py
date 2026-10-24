"""Extraction pipeline v1 (Milestone 2, Task #4).

Loads the cleaned NEISS data, preprocesses the narratives, runs GLiNER, and
writes one extracted object per narrative to a CSV.

Run from the repo root (see data/README.md for how to build the input file):

    python -m src.extract --sample 1000
    python -m src.extract --text-field normalized --threshold 0.4
    python -m src.extract                      # all rows, slow on a CPU

What to swap, and where:
    prompt    ExtractionConfig.labels (the entity labels GLiNER is asked for),
              .threshold, .model_name
    input     ExtractionConfig.text_field ("event" or "normalized"), .lowercase
    rules     ExtractionConfig.keep_entity (drop unwanted spans) and
              .choose_entity (pick one span out of several)
Cleaning rules live in src/preprocess.py and are shared with the EDA notebook.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence

import pandas as pd

try:  # python -m src.extract
    from src.preprocess import preprocess, split_diagnosis
except ImportError:  # python src/extract.py
    from preprocess import preprocess, split_diagnosis

DEFAULT_INPUT = Path("data/processed/neiss2025_clean.csv")
DEFAULT_OUTPUT = Path("data/processed/extractions_v1.csv")

# Same model, labels and threshold as reports/model_logic_plan.md and the
# Task #5 sample notebook, so results are comparable.
DEFAULT_MODEL = "urchade/gliner_medium-v2.1"
DEFAULT_LABELS = ("object", "tool", "furniture", "sports equipment", "structure")
DEFAULT_THRESHOLD = 0.30
FALLBACK = "OTHER"

# Columns copied to the output when they exist, so predictions can be compared
# with the NEISS ground truth later (Tasks 6 and 7).
ID_COLUMN = "case_number"
PASSTHROUGH_COLUMNS = [
    "product_code",
    "product_label",
    "product_2_code",
    "product_3_code",
    "vague_label",
    "rare_class",
    "redacted",
    "psu",
]

# Starter rule: spans that are people or body parts, never the product.
# Edit freely; this is meant to grow as error analysis (Task 8) finds patterns.
DROP_SPANS = frozenset(
    {
        "head", "face", "eye", "eyes", "nose", "mouth", "ear", "neck", "arm", "hand",
        "finger", "thumb", "leg", "knee", "ankle", "foot", "toe", "back", "chest",
        "shoulder", "wrist", "elbow", "hip",
        "brother", "sister", "mother", "father", "friend", "son", "daughter", "patient",
    }
)


class Extraction(NamedTuple):
    text: Optional[str]
    entity_type: Optional[str]
    confidence: Optional[float]


def default_keep_entity(entity: dict) -> bool:
    """Rule 1: drop spans that are people or body parts."""
    return entity["text"].strip().lower() not in DROP_SPANS


def default_choose_entity(entities: Sequence[dict]) -> dict:
    """Rule 2: when several spans survive, keep the highest-confidence one."""
    return max(entities, key=lambda e: float(e["score"]))


@dataclass(frozen=True)
class ExtractionConfig:
    model_name: str = DEFAULT_MODEL
    labels: tuple = DEFAULT_LABELS
    threshold: float = DEFAULT_THRESHOLD
    text_field: str = "event"  # "event" = narrative without the DX section
    lowercase: bool = False  # test whether mixed case helps GLiNER
    batch_size: int = 32
    fallback: str = FALLBACK
    keep_entity: Callable[[dict], bool] = default_keep_entity
    choose_entity: Callable[[Sequence[dict]], dict] = default_choose_entity


def prepare_text(narratives: pd.Series, config: ExtractionConfig) -> pd.DataFrame:
    """Clean raw narratives with src/preprocess.py and pick the model input.

    Returns narrative_normalized, event_text and model_text (what GLiNER sees).
    The DX split runs before lowercasing because the DX marker is matched in
    uppercase.
    """
    if config.text_field not in {"event", "normalized"}:
        raise ValueError("text_field must be 'event' or 'normalized'")

    normalized = preprocess(narratives.fillna("").astype(str))
    event = split_diagnosis(normalized)["event"]
    model_text = event if config.text_field == "event" else normalized
    if config.lowercase:
        model_text = model_text.str.lower()
    return pd.DataFrame(
        {"narrative_normalized": normalized, "event_text": event, "model_text": model_text}
    )


def load_model(config: ExtractionConfig):
    """Load GLiNER. Downloads the weights on first use."""
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise SystemExit("gliner is not installed. Run: pip install -r requirements.txt") from exc
    return GLiNER.from_pretrained(config.model_name)


def predict_entities(model, texts: list, config: ExtractionConfig, progress: bool = False) -> list:
    """Run the model in batches. Returns one list of entity dicts per text."""
    labels = list(config.labels)
    batch_fn = getattr(model, "batch_predict_entities", None)
    results: list = []
    for start in range(0, len(texts), config.batch_size):
        chunk = texts[start : start + config.batch_size]
        if batch_fn is not None:
            results.extend(batch_fn(chunk, labels, threshold=config.threshold))
        else:
            results.extend(
                model.predict_entities(t, labels, threshold=config.threshold) for t in chunk
            )
        if progress and (start // config.batch_size) % 50 == 0:
            print(f"  {min(start + len(chunk), len(texts)):,} / {len(texts):,}")
    return results


def extract_objects(
    texts: Sequence[str], model, config: ExtractionConfig, progress: bool = False
) -> list:
    """One Extraction per text. Blank text, or no surviving span, gives the fallback."""
    texts = list(texts)
    out = [Extraction(config.fallback, None, None)] * len(texts)
    todo = [i for i, t in enumerate(texts) if t and t.strip()]
    entity_lists = predict_entities(model, [texts[i] for i in todo], config, progress)

    for i, entities in zip(todo, entity_lists):
        kept = [e for e in entities if config.keep_entity(e)]
        if kept:
            best = config.choose_entity(kept)
            out[i] = Extraction(best["text"].strip(), best["label"], float(best["score"]))
    return out


def run(
    df: pd.DataFrame,
    model,
    config: ExtractionConfig = ExtractionConfig(),
    narrative_column: str = "narrative",
    progress: bool = False,
) -> pd.DataFrame:
    """Full pipeline on a DataFrame with a raw narrative column. One row in, one row out."""
    if narrative_column not in df.columns:
        raise KeyError(f"column '{narrative_column}' not found; columns are {list(df.columns)}")

    prepared = prepare_text(df[narrative_column], config)
    extractions = extract_objects(prepared["model_text"].tolist(), model, config, progress)

    out = pd.DataFrame(index=df.index)
    if ID_COLUMN in df.columns:
        out[ID_COLUMN] = df[ID_COLUMN]
    out["narrative"] = df[narrative_column]
    out["narrative_normalized"] = prepared["narrative_normalized"]
    out["event_text"] = prepared["event_text"]
    out["extracted_object"] = [e.text for e in extractions]
    out["extracted_entity_type"] = [e.entity_type for e in extractions]
    out["extraction_confidence"] = [e.confidence for e in extractions]
    for col in PASSTHROUGH_COLUMNS:
        if col in df.columns:
            out[col] = df[col]
    return out.reset_index(drop=True)


def load_clean(path: Path, sample: Optional[int] = None, seed: int = 100) -> pd.DataFrame:
    """Load the cleaned NEISS CSV, optionally a reproducible random sample."""
    if not Path(path).exists():
        raise SystemExit(
            f"{path} not found. Build it by running notebooks/milestone1_eda.ipynb "
            "(see data/README.md, step 3)."
        )
    df = pd.read_csv(path, low_memory=False)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=seed)
    return df.reset_index(drop=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEISS object extraction pipeline v1")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--narrative-column", default="narrative")
    p.add_argument("--sample", type=int, default=None, help="random sample of N rows (default: all)")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--labels", default=",".join(DEFAULT_LABELS), help="comma-separated entity labels")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--text-field", choices=["event", "normalized"], default="event")
    p.add_argument("--lowercase", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = ExtractionConfig(
        model_name=args.model,
        labels=tuple(s.strip() for s in args.labels.split(",") if s.strip()),
        threshold=args.threshold,
        text_field=args.text_field,
        lowercase=args.lowercase,
        batch_size=args.batch_size,
    )

    df = load_clean(args.input, args.sample, args.seed)
    print(f"Loaded {len(df):,} narratives from {args.input}")
    if args.sample is None and len(df) > 20_000:
        print("Running on all rows. This is slow on a CPU; use --sample 1000 to test first.")

    model = load_model(config)
    result = run(df, model, config, args.narrative_column, progress=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    fallback = (result["extracted_object"] == config.fallback).sum()
    print(f"Wrote {len(result):,} rows to {args.output}")
    print(f"{config.fallback} (no object found): {fallback:,} ({fallback / len(result):.1%})")


if __name__ == "__main__":
    main()
