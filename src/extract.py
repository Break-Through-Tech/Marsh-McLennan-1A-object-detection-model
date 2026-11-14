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
    rules     ExtractionConfig.clean_span (tidy a span), .keep_entity (drop unwanted spans),
              .choose_entity (pick one span out of several) and
              .fallback_rule (what to output when no span survives: keyword, then floor)
Cleaning rules live in src/preprocess.py and are shared with the EDA notebook.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence

import pandas as pd

try:  # python -m src.extract
    from src.preprocess import preprocess, split_diagnosis
except ImportError:  # python src/extract.py
    from preprocess import preprocess, split_diagnosis

DEFAULT_INPUT = Path("data/processed/neiss2025_clean.csv")
DEFAULT_OUTPUT = Path("data/processed/extractions_v3.csv")

# Same model, labels and threshold as reports/model_logic_plan.md and the
# Task #5 sample notebook, so results are comparable.
DEFAULT_MODEL = "urchade/gliner_medium-v2.1"
DEFAULT_LABELS = ("object", "tool", "furniture", "sports equipment", "structure", "sport")
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

# Rule 1 (Task #9): spans that are never the product. Exact match on the whole
# span, so "car seat" or "bed frame" are kept. Edit freely; the lists grow as
# error analysis finds patterns.
PEOPLE = {"brother", "sister", "mother", "father", "friend", "son", "daughter", "patient"}
BODY_PARTS = {
    "head", "face", "eye", "eyes", "nose", "mouth", "ear", "neck", "arm", "hand",
    "finger", "thumb", "leg", "knee", "ankle", "foot", "toe", "back", "chest",
    "shoulder", "wrist", "elbow", "hip", "tailbone", "forehead", "scalp", "chin",
    "lip", "tooth", "teeth", "rib", "ribs", "spine", "skull", "thigh", "calf",
    "heel", "palm", "fingers", "toes", "lower back", "upper back", "buttocks",
}
PLACES = {
    "bathroom", "kitchen", "bedroom", "room", "intersection", "pond", "road",
    "street", "yard", "home", "house", "hospital", "school", "store", "outside",
}
OTHER_PARTY_VEHICLES = {"car", "truck", "vehicle", "motor vehicle", "suv", "van", "bus"}
IMPLANTED_DEVICES = {"prosthetic device", "prosthesis", "pacemaker", "implant"}
DROP_SPANS = frozenset(PEOPLE | BODY_PARTS | PLACES | OTHER_PARTY_VEHICLES | IMPLANTED_DEVICES)

# Rule 3 (Task #9): labeling rule 4 says a fall with no named object is a floor.
FLOOR = "FLOOR"
FLOOR_ENTITY_TYPE = "rule: floor default"
_FLOOR_WORD = re.compile(r"\bFLOORS?\b", re.IGNORECASE)
_FALL_WORDS = re.compile(r"\b(FELL|FALL|FALLS|FALLING|FALLEN|TRIPPED|SLIPPED|SLIP)\b", re.IGNORECASE)
_FELL_ASLEEP = re.compile(r"\bFELL ASLEEP\b", re.IGNORECASE)
# Outdoor ground is not a product (labeling rule 4), and crashes are not floor falls.
_NOT_A_FLOOR_FALL = re.compile(
    r"\b(OUTSIDE|OUTDOORS?|SIDEWALK|DRIVEWAY|ROAD|STREET|INTERSECTION|PARKING LOT|GRASS|DIRT|"
    r"ICE|SNOW|CURB|GROUND|CONCRETE|PAVEMENT|ASPHALT|CAR|TRUCK|VEHICLE|CRASHED|COLLISION|MVA)\b",
    re.IGNORECASE,
)


# Rule 4 (Task #2): tidy every span before the stoplist and the scorer see it.
# Leading and trailing words that describe the action, not the product, are
# dropped ("HANDLING KNIFE" -> "KNIFE", "BASKETBALL GAME" -> "BASKETBALL"), and
# near-miss spellings of common objects are fixed ("WHELLCHAIR" -> "WHEELCHAIR").
LEADING_WORDS = {"a", "an", "the", "his", "her", "their", "my", "some",
                 "using", "handling", "holding", "carrying",
                 "right", "left", "rt", "lt", "upper", "lower"}
TRAILING_WORDS = {"game", "games", "practice", "match", "class", "accident", "injury"}
PROTECTED_SPANS = {"video game", "board game", "card game", "game"}  # real products
TYPO_VOCAB = (
    "wheelchair", "trampoline", "skateboard", "bicycle", "scooter", "basketball",
    "football", "baseball", "softball", "treadmill", "recliner", "dresser",
    "refrigerator", "stroller", "nightstand", "bookshelf", "bathtub", "mattress",
    "staircase", "furniture", "hoverboard", "lawnmower", "bookcase", "microwave",
)
TYPO_CUTOFF = 0.88


def fix_typo(word: str) -> str:
    """Return the closest common object word when the spelling is a near miss."""
    low = word.lower()
    if len(low) < 7 or low in TYPO_VOCAB:
        return word
    match = difflib.get_close_matches(low, TYPO_VOCAB, n=1, cutoff=TYPO_CUTOFF)
    if not match:
        return word
    return match[0].upper() if word.isupper() else match[0]


def default_clean_span(text: str) -> str:
    """Rule 4: drop action words around the product and fix near-miss spellings."""
    words = text.strip().split()
    if not words:
        return text
    while len(words) > 1 and words[0].lower() in LEADING_WORDS:
        words = words[1:]

    while len(words) > 1 and words[-1].lower() in TRAILING_WORDS:
        if " ".join(words).lower() in PROTECTED_SPANS:
            break
        words = words[:-1]


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


def floor_for_falls(text: str) -> Optional["Extraction"]:
    """Rule 3: when no object was found, a described fall defaults to floor.

    Fires when the text mentions a floor, or describes a fall that is not
    outdoors or a crash. It never fires on redacted text (a *** may hide the
    real object).
    """
    if "*" in text:
        return None
    if _FLOOR_WORD.search(text):
        return Extraction(FLOOR, FLOOR_ENTITY_TYPE, None)
    if _FALL_WORDS.search(_FELL_ASLEEP.sub("", text)) and not _NOT_A_FLOOR_FALL.search(text):
        return Extraction(FLOOR, FLOOR_ENTITY_TYPE, None)
    return None


# Rule 5 (Task #2): when the model finds nothing, look for a common household
# object by name. These came from the Task #9 check: of 23 falls where the floor
# rule disagreed with NEISS, 17 named a bed (5), steps (4), a shower or tub (5),
# a scooter, a bike or a dresser that the model missed. The earliest match in
# the text wins.
KEYWORD_ENTITY_TYPE = "rule: keyword"
KEYWORD_OBJECTS = (
    ("BED", re.compile(r"\bBED\b", re.IGNORECASE)),
    ("STAIRS", re.compile(r"\b(STAIRS|STAIRWAY|STAIRCASE|STEPS?)\b", re.IGNORECASE)),
    ("SHOWER", re.compile(r"\bSHOWER\b", re.IGNORECASE)),
    ("BATHTUB", re.compile(r"\b(BATHTUB|TUB|BATH)\b", re.IGNORECASE)),
    ("SCOOTER", re.compile(r"\bSCOOTERS?\b", re.IGNORECASE)),
    ("BICYCLE", re.compile(r"\b(BICYCLE|(?<!DIRT )BIKE)\b", re.IGNORECASE)),
    ("DRESSER", re.compile(r"\bDRESSER\b", re.IGNORECASE)),
)


def keyword_object(text: str) -> Optional["Extraction"]:
    """Rule 5: name the earliest common household object found in the text."""
    best = None
    for name, pattern in KEYWORD_OBJECTS:
        m = pattern.search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), name)
    if best is None:
        return None
    return Extraction(best[1], KEYWORD_ENTITY_TYPE, None)


def keyword_then_floor(text: str) -> Optional["Extraction"]:
    """Default fallback: a named household object first, then the floor default."""
    return keyword_object(text) or floor_for_falls(text)


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
    # Tidies each span's text before keep_entity sees it; None switches it off.
    clean_span: Optional[Callable[[str], str]] = default_clean_span
    choose_entity: Callable[[Sequence[dict]], dict] = default_choose_entity
    # Called with the model text when no span survives; return an Extraction or None.
    # Set to None to switch the keyword and floor rules off.
    fallback_rule: Optional[Callable[[str], Optional["Extraction"]]] = keyword_then_floor


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
    """One Extraction per text. Blank text gives the fallback; no surviving span tries
    config.fallback_rule, then the fallback."""
    texts = list(texts)
    out = [Extraction(config.fallback, None, None)] * len(texts)
    todo = [i for i, t in enumerate(texts) if t and t.strip()]
    entity_lists = predict_entities(model, [texts[i] for i in todo], config, progress)

    for i, entities in zip(todo, entity_lists):
        if config.clean_span is not None:
            entities = [{**e, "text": config.clean_span(e["text"])} for e in entities]
        kept = [e for e in entities if config.keep_entity(e)]
        if kept:
            best = config.choose_entity(kept)
            out[i] = Extraction(best["text"].strip(), best["label"], float(best["score"]))
        elif config.fallback_rule is not None:
            ruled = config.fallback_rule(texts[i])
            if ruled is not None:
                out[i] = ruled
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


def summarize_run(
    result: pd.DataFrame,
    input_rows: int,
    elapsed_seconds: float,
    config: ExtractionConfig,
    top_n: int = 20,
) -> dict:
    """Run statistics: row counts, runtime, fallback rate, failed rows, top objects.

    A row "failed" when its extracted_object is missing or blank. A fallback
    row (OTHER) is a valid output, counted separately.
    """
    obj = result["extracted_object"]
    failed = obj.isna() | (obj.astype(str).str.strip() == "")
    fallback = int((obj == config.fallback).sum())
    n = len(result)
    return {
        "input_rows": int(input_rows),
        "output_rows": int(n),
        "runtime_seconds": round(float(elapsed_seconds), 2),
        "rows_per_second": round(n / elapsed_seconds, 2) if elapsed_seconds > 0 else None,
        "fallback_label": config.fallback,
        "fallback_rows": fallback,
        "fallback_pct": round(100 * fallback / n, 2) if n else None,
        "floor_rule_rows": int((result["extracted_entity_type"] == FLOOR_ENTITY_TYPE).sum()),
        "failed_rows": int(failed.sum()),
        "failed_case_numbers": (
            result.loc[failed, ID_COLUMN].head(100).tolist() if ID_COLUMN in result else []
        ),
        "top_objects": {str(k): int(v) for k, v in obj[~failed].value_counts().head(top_n).items()},
        "config": {
            "model": config.model_name,
            "labels": list(config.labels),
            "threshold": config.threshold,
            "text_field": config.text_field,
            "batch_size": config.batch_size,
        },
    }


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
    p.add_argument("--log", type=Path, default=None, help="write run statistics (runtime, fallback rate, failures) to this JSON file")
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
    start = time.perf_counter()
    result = run(df, model, config, args.narrative_column, progress=True)
    elapsed = time.perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    fallback = (result["extracted_object"] == config.fallback).sum()
    print(f"Wrote {len(result):,} rows to {args.output}")
    print(f"{config.fallback} (no object found): {fallback:,} ({fallback / len(result):.1%})")
    print(f"Runtime: {elapsed:,.1f} s")
    if args.log is not None:
        stats = summarize_run(result, len(df), elapsed, config)
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Wrote run log to {args.log} ({stats['failed_rows']:,} failed rows)")


if __name__ == "__main__":
    main()
