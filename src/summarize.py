"""Summary statistics for extracted objects (Milestone 2, Task #11).

Joins the extraction output (Task #10) with the NEISS fields in the cleaned
dataset, counts and ranks the extracted objects, and summarizes them by age
group, body part, diagnosis, disposition, location, fire involvement, alcohol
and drug. Writes two CSVs and a markdown report to reports/.

Run from the repo root:

    python -m src.summarize
    python -m src.summarize --extractions data/processed/extractions_v2.csv --top 25
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

ID_COLUMN = "case_number"
OBJECT_COLUMN = "extracted_object"
FALLBACK = "OTHER"

DEFAULT_EXTRACTIONS = Path("data/processed/extractions_v2.csv")
DEFAULT_CLEAN = Path("data/processed/neiss2025_clean.csv")
DEFAULT_FMT = Path("data/raw/neiss2025 1(NEISS_FMT).csv")
DEFAULT_OUT_DIR = Path("reports")

# cleaned-dataset column -> NEISS_FMT format name (None = no codebook decode)
FIELDS: Dict[str, Optional[str]] = {
    "age_group": None,
    "body_part": "BDYPT",
    "diagnosis": "DIAG",
    "disposition": "DISP",
    "location": "LOC",
    "fire_involvement": "FIRE",
    "alcohol": "ALC_DRUG",
    "drug": "ALC_DRUG",
}

AGE_BINS = [-0.1, 4, 14, 24, 44, 64, 200]
AGE_LABELS = ["0-4", "5-14", "15-24", "25-44", "45-64", "65+"]
UNKNOWN = "Unknown"


def age_group(age_years: pd.Series) -> pd.Series:
    """Bucket age in years; missing ages stay missing (reported as Unknown)."""
    return pd.cut(pd.to_numeric(age_years, errors="coerce"), AGE_BINS, labels=AGE_LABELS).astype(object)


def load_codebook(fmt: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """NEISS_FMT as {format name: DataFrame(start, end, label)}."""
    fmt = fmt.copy()
    fmt.columns = ["format", "start", "end", "label"]
    for col in ("format", "label"):
        fmt[col] = fmt[col].astype(str).str.strip()
    fmt["start"] = pd.to_numeric(fmt["start"], errors="coerce")
    fmt["end"] = pd.to_numeric(fmt["end"], errors="coerce")
    return {name: g[["start", "end", "label"]] for name, g in fmt.groupby("format")}


def decode(codes: pd.Series, table: pd.DataFrame) -> pd.Series:
    """Map numeric NEISS codes to labels using the (start, end) ranges. Blank stays blank."""
    codes = pd.to_numeric(codes, errors="coerce")
    out = pd.Series(pd.NA, index=codes.index, dtype=object)
    for start, end, label in table.itertuples(index=False):
        out[(codes >= start) & (codes <= end)] = label
    return out


def build_frame(extractions: pd.DataFrame, clean: pd.DataFrame, fmt: Optional[pd.DataFrame]) -> pd.DataFrame:
    """One row per case: extracted object plus decoded NEISS fields."""
    for name, df in (("extractions", extractions), ("clean", clean)):
        if ID_COLUMN not in df.columns:
            raise KeyError(f"{name} has no '{ID_COLUMN}' column")
    if OBJECT_COLUMN not in extractions.columns:
        raise KeyError(f"extractions has no '{OBJECT_COLUMN}' column")

    wanted = [c for c in FIELDS if c != "age_group" and c in clean.columns]
    extra = ["age_years"] if "age_years" in clean.columns else []
    # Take only the NEISS fields from the cleaned file; extractions may already carry some columns.
    base = extractions[[ID_COLUMN, OBJECT_COLUMN]].drop_duplicates(ID_COLUMN)
    frame = base.merge(clean[[ID_COLUMN] + wanted + extra].drop_duplicates(ID_COLUMN), on=ID_COLUMN, how="left")
    if len(frame) != len(base):
        raise ValueError("join changed the row count")

    codebook = load_codebook(fmt) if fmt is not None else {}
    for col in wanted:
        table = codebook.get(FIELDS[col])
        frame[col] = decode(frame[col], table) if table is not None else frame[col].astype(object)
    if extra:
        frame["age_group"] = age_group(frame.pop("age_years"))
    return frame


def rank_objects(frame: pd.DataFrame, fallback: str = FALLBACK) -> pd.DataFrame:
    """Rank every extracted object. Columns: rank, object, count, pct, cum_pct.

    Percentages are of all rows. `rank` skips the fallback (it is not an
    object) so rank 1 is the most common real object; the fallback row has
    rank blank.
    """
    counts = frame[OBJECT_COLUMN].fillna(fallback).value_counts()
    out = counts.rename_axis("object").reset_index(name="count")
    out["pct"] = 100 * out["count"] / out["count"].sum()
    out["cum_pct"] = out["pct"].cumsum()
    is_real = out["object"] != fallback
    out["rank"] = pd.NA
    out.loc[is_real, "rank"] = range(1, int(is_real.sum()) + 1)
    out["rank"] = out["rank"].astype("Int64")
    return out[["rank", "object", "count", "pct", "cum_pct"]]


def objects_by_field(frame: pd.DataFrame, fields: Optional[list] = None, top_objects: int = 25) -> pd.DataFrame:
    """Long table of the top objects crossed with each NEISS field.

    Columns: field, value, object, count, pct_of_object (share of the object's
    cases with this value), pct_of_value (share of this value's cases that are
    this object). Missing values are grouped as Unknown. The fallback is excluded.
    """
    fields = [f for f in (fields or FIELDS) if f in frame.columns]
    real = frame[frame[OBJECT_COLUMN].notna() & (frame[OBJECT_COLUMN] != FALLBACK)]
    top = real[OBJECT_COLUMN].value_counts().head(top_objects).index
    parts = []
    for field in fields:
        values = real[field].fillna(UNKNOWN).astype(str)
        value_totals = values.value_counts()
        object_totals = real[OBJECT_COLUMN].value_counts()
        g = (
            pd.DataFrame({"object": real[OBJECT_COLUMN], "value": values})
            .loc[real[OBJECT_COLUMN].isin(top)]
            .groupby(["object", "value"]).size().rename("count").reset_index()
        )
        g["field"] = field
        g["pct_of_object"] = 100 * g["count"] / g["object"].map(object_totals)
        g["pct_of_value"] = 100 * g["count"] / g["value"].map(value_totals)
        parts.append(g)
    if not parts:
        return pd.DataFrame(columns=["field", "value", "object", "count", "pct_of_object", "pct_of_value"])
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["field", "object", "count"], ascending=[True, True, False])
    return out[["field", "value", "object", "count", "pct_of_object", "pct_of_value"]].reset_index(drop=True)


def top_objects_per_value(frame: pd.DataFrame, field: str, n: int = 3, min_cases: int = 500) -> pd.DataFrame:
    """For each value of a field, its n most common objects (fallback excluded)."""
    real = frame[frame[OBJECT_COLUMN].notna() & (frame[OBJECT_COLUMN] != FALLBACK)].copy()
    real["value"] = real[field].fillna(UNKNOWN).astype(str)
    sizes = real["value"].value_counts()
    keep = sizes[sizes >= min_cases].index
    g = real[real["value"].isin(keep)].groupby(["value", OBJECT_COLUMN]).size().rename("count").reset_index()
    g["pct_of_value"] = 100 * g["count"] / g["value"].map(sizes)
    g = g.sort_values(["value", "count"], ascending=[True, False]).groupby("value").head(n)
    return g.rename(columns={OBJECT_COLUMN: "object"}).reset_index(drop=True)


def coverage(frame: pd.DataFrame) -> dict:
    """Headline numbers: rows, distinct objects, fallback share, top-10/top-50 share."""
    n = len(frame)
    obj = frame[OBJECT_COLUMN].fillna(FALLBACK)
    real = obj[obj != FALLBACK]
    vc = real.value_counts()
    return {
        "rows": n,
        "fallback_rows": int((obj == FALLBACK).sum()),
        "fallback_pct": 100 * (obj == FALLBACK).sum() / n if n else 0.0,
        "distinct_objects": int(vc.size),
        "top10_pct_of_objects": 100 * vc.head(10).sum() / len(real) if len(real) else 0.0,
        "top50_pct_of_objects": 100 * vc.head(50).sum() / len(real) if len(real) else 0.0,
        "singletons": int((vc == 1).sum()),
    }


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = [f"{v:,.1f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) else str(v)) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(frame: pd.DataFrame, ranking: pd.DataFrame, by_field: pd.DataFrame, top: int) -> str:
    cov = coverage(frame)
    shown = ranking.head(top + int((ranking["object"] == FALLBACK).head(top).sum())).copy()
    shown["rank"] = shown["rank"].astype(object).where(shown["rank"].notna(), "-")
    shown["pct"] = shown["pct"].round(2)
    shown["cum_pct"] = shown["cum_pct"].round(2)
    lines = [
        "# Extracted-object summary statistics",
        "",
        "Marsh McLennan 1A, Break Through Tech AI Studio, Fall 2026",
        "Milestone 2, Task #11",
        "",
        "Counts and ranks the objects extracted in Task #10 and summarizes them by NEISS fields. "
        "Generated by `python -m src.summarize`; full tables are in `reports/task11_object_ranking.csv` "
        "and `reports/task11_object_by_field.csv`.",
        "",
        "## Overview",
        "",
        f"- Cases: {cov['rows']:,}",
        f"- Fallback ({FALLBACK}, no object found): {cov['fallback_rows']:,} ({cov['fallback_pct']:.2f}%)",
        f"- Distinct extracted objects: {cov['distinct_objects']:,} ({cov['singletons']:,} appear once)",
        f"- Top 10 objects cover {cov['top10_pct_of_objects']:.1f}% of extracted objects; top 50 cover {cov['top50_pct_of_objects']:.1f}%",
        "",
        f"## Top {top} objects",
        "",
        "Percentages are of all cases. Rank ignores the fallback.",
        "",
        md_table(shown),
        "",
    ]
    for field in [f for f in FIELDS if f in frame.columns]:
        tv = top_objects_per_value(frame, field)
        if tv.empty:
            continue
        tv = tv.assign(pct_of_value=tv["pct_of_value"].round(1))
        lines += [
            f"## Most common objects by {field.replace('_', ' ')}",
            "",
            "Top 3 objects for each value with at least 500 cases (fallback excluded).",
            "",
            md_table(tv[["value", "object", "count", "pct_of_value"]]),
            "",
        ]
    lines += [
        "## Limitations",
        "",
        "- Objects are model predictions, not ground truth. Task #9 measured strict F1 0.57 and lenient F1 0.76 on a 25-case sample, so counts carry that error.",
        "- FLOOR is inflated by the Task #9 default rule (a fall with no named object becomes FLOOR).",
        f"- {FALLBACK} rows are excluded from object rankings and field breakdowns; they include redacted narratives.",
        "- NEISS case weights are not applied, so counts are ED visits in the sample, not national estimates.",
        "- Unknown NEISS codes were blanked in the cleaned dataset and appear as Unknown here.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summary statistics for extracted objects")
    p.add_argument("--extractions", type=Path, default=DEFAULT_EXTRACTIONS)
    p.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    p.add_argument("--fmt", type=Path, default=DEFAULT_FMT, help="NEISS_FMT codebook CSV")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--top", type=int, default=25, help="objects shown in the report and field breakdown")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    for path in (args.extractions, args.clean):
        if not path.exists():
            raise SystemExit(f"{path} not found. See reports/task10_full_run.md and data/README.md.")
    fmt = pd.read_csv(args.fmt) if args.fmt.exists() else None
    if fmt is None:
        print(f"Warning: {args.fmt} not found; NEISS codes will not be decoded.")

    extractions = pd.read_csv(args.extractions, usecols=[ID_COLUMN, OBJECT_COLUMN], low_memory=False)
    clean = pd.read_csv(args.clean, low_memory=False)
    frame = build_frame(extractions, clean, fmt)
    ranking = rank_objects(frame)
    by_field = objects_by_field(frame, top_objects=args.top)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.out_dir / "task11_object_ranking.csv", index=False)
    by_field.to_csv(args.out_dir / "task11_object_by_field.csv", index=False)
    (args.out_dir / "task11_summary_statistics.md").write_text(
        build_report(frame, ranking, by_field, args.top), encoding="utf-8"
    )
    distinct = int((ranking["object"] != FALLBACK).sum())
    print(f"Wrote 3 files to {args.out_dir} ({len(frame):,} cases, {distinct:,} distinct objects excluding {FALLBACK})")


if __name__ == "__main__":
    main()
