"""Check the rule-made predictions against NEISS product labels (Milestone 3, Task #2).

The floor and keyword rules fire when the model finds nothing. NEISS has no
hand labels for the full data, but each row has a coder-assigned product label.
This script counts how often a rule's answer agrees with it. NEISS lists one
primary product per case, so agreement is a rough check, not an accuracy score.

Run from the repo root after an extraction run:

    python -m src.check_rules --input data/processed/extractions_v3.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Rule answer -> text that appears in the matching NEISS product label.
NEISS_LABEL_PARTS = {
    "FLOOR": "FLOORS OR FLOORING",
    "BED": "BEDS OR BEDFRAMES",
    "STAIRS": "STAIRS OR STEPS",
    "SHOWER": "BATHTUBS OR SHOWERS",
    "BATHTUB": "BATHTUBS OR SHOWERS",
    "SCOOTER": "SCOOTERS",
    "BICYCLE": "BICYCLES",
    "DRESSER": "DESKS, CHESTS, BUREAUS",
}


def rule_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (rule, answer): how many times it fired and how often NEISS agrees."""
    rules = df[df["extracted_entity_type"].fillna("").str.startswith("rule:")].copy()
    label = rules["product_label"].fillna("").str.upper()
    expected = rules["extracted_object"].map(NEISS_LABEL_PARTS).fillna("\0")
    rules["agrees"] = [part in lab for part, lab in zip(expected, label)]
    out = (rules.groupby(["extracted_entity_type", "extracted_object"])["agrees"]
           .agg(fired="size", agree="sum").reset_index())
    out["agree_pct"] = (100 * out["agree"] / out["fired"]).round(1)
    return out.sort_values("fired", ascending=False).reset_index(drop=True)


def main(argv=None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description="Compare rule-made predictions with NEISS labels")
    p.add_argument("--input", type=Path, required=True)
    args = p.parse_args(argv)
    df = pd.read_csv(args.input, low_memory=False)
    table = rule_agreement(df)

    n = len(df)
    fired, agree = int(table["fired"].sum()), int(table["agree"].sum())
    print(f"Rows: {n:,}")
    print(f"OTHER rows: {(df['extracted_object'] == 'OTHER').sum():,}")
    print(f"Rule-made predictions: {fired:,} ({100 * fired / n:.1f}% of rows)")
    if fired:
        print(f"Agree with the NEISS product label: {agree:,} ({100 * agree / fired:.1f}%)")
    print()
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    main()
