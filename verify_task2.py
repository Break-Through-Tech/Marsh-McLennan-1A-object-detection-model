"""Verify the saved Task #2 results without rerunning GLiNER.

Run from the repository root after the Task #2 files have been copied:
    python verify_task2.py
"""

from pathlib import Path
import pandas as pd
from src import extract


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def metrics(frame: pd.DataFrame, rule: str) -> tuple[int, int, int, float, float, float]:
    answered = frame["extracted_object"].fillna("OTHER").astype(str).str.strip().str.upper() != "OTHER"
    correct = frame[f"{rule}_correct"].astype(str).str.casefold().isin(["true", "1"])
    tp = int((answered & correct).sum())
    fp = int((answered & ~correct).sum())
    fn = len(frame) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1


def main() -> None:
    require(extract.DEFAULT_LABELS == (
        "object", "tool", "furniture", "sports equipment", "structure", "sport"
    ), "src/extract.py does not use the selected +sport label set")
    config = extract.ExtractionConfig()
    require(config.text_field == "event" and config.lowercase is False,
            "Final model preprocessing defaults differ from the reported settings")
    require(abs(config.threshold - 0.30) < 1e-8, "The final model threshold should be 0.30")
    require(config.clean_span is extract.default_clean_span, "Span cleaning is not enabled")
    require(config.fallback_rule is extract.keyword_then_floor, "Keyword-plus-floor fallback is not enabled")
    require(str(extract.DEFAULT_OUTPUT).endswith("extractions_v3.csv"),
            "Default output would overwrite an older version; should end in extractions_v3.csv")

    grid_path = Path("reports/task2_tuning_grid.csv")
    detail_path = Path("reports/task2_eval_details.csv")
    best_path = Path("reports/task2_tuning_best_details.csv")
    for file in (grid_path, detail_path, best_path):
        require(file.exists(), f"Required saved result is missing: {file}")

    grid = pd.read_csv(grid_path)
    require(len(grid) == 256, f"Expected 256 tuning configurations, got {len(grid)}")
    selection = grid[
        grid["labels"].eq("+sport") &
        grid["text_field"].eq("event") &
        grid["lowercase"].astype(str).str.casefold().eq("false") &
        (grid["threshold"].astype(float) - 0.30).abs().lt(1e-8) &
        grid["clean_span"].eq("on") &
        grid["fallback"].eq("keyword+floor")
    ]
    require(len(selection) == 1, "Could not identify the final chosen configuration in the grid")
    require(round(float(selection.iloc[0]["strict_f1"]), 2) == 0.76,
            "Grid strict F1 differs from the reported final strict F1")
    require(round(float(selection.iloc[0]["lenient_f1"]), 2) == 0.89,
            "Grid lenient F1 differs from the reported final lenient F1")

    details = pd.read_csv(detail_path, dtype={"case_number": str})
    require(len(details) == 25, f"Expected 25 labeled validation rows, got {len(details)}")
    require(details["case_number"].is_unique, "Duplicate case IDs in evaluation details")
    strict = metrics(details, "strict")
    lenient = metrics(details, "lenient")
    require(strict[:3] == (17, 3, 8), f"Unexpected strict TP/FP/FN: {strict[:3]}")
    require(lenient[:3] == (20, 0, 5), f"Unexpected lenient TP/FP/FN: {lenient[:3]}")

    best = pd.read_csv(best_path, dtype={"case_number": str})
    require(len(best) == 25 and best["case_number"].is_unique,
            "The best-grid configuration's case-level CSV should contain 25 distinct rows")
    require(metrics(best, "strict")[:3] == (19, 4, 6),
            "Best-grid strict counts do not match the tuning output")

    print("PASS: final extraction defaults match Task #2 report")
    print("PASS: 256 configurations, and the final settings match the saved tuning grid")
    print("PASS: saved final evaluation has 25 unique cases")
    print(f"PASS: strict  TP={strict[0]} FP={strict[1]} FN={strict[2]} "
          f"P={strict[3]:.2f} R={strict[4]:.2f} F1={strict[5]:.2f}")
    print(f"PASS: lenient TP={lenient[0]} FP={lenient[1]} FN={lenient[2]} "
          f"P={lenient[3]:.2f} R={lenient[4]:.2f} F1={lenient[5]:.2f}")
    print("PASS: best-grid case-level results match the reported TP/FP/FN")
    print("NOTE: these checks do not rerun GLiNER or establish held-out generalization.")


if __name__ == "__main__":
    main()
