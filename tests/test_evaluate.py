"""Tests for src/evaluate.py. No model download needed:  python -m pytest tests"""

import pandas as pd

from src import evaluate
from src.evaluate import gold_names, lenient_match, metrics, score, strict_match


def test_gold_names_split_slashes_and_drop_activity():
    assert gold_names("stairs/steps") == ["stair", "step"]
    assert gold_names("basketball (activity)") == ["basketball"]
    assert "floor" in gold_names("floor (tile)")


def test_strict_and_lenient_matching():
    assert strict_match("STEPS", "stairs/steps")
    assert strict_match("knives", "knife")
    assert strict_match("Skateboard", "skateboard")
    assert not strict_match("tile floor", "floor (tile)")
    assert lenient_match("tile floor", "floor (tile)")
    assert not lenient_match("car", "bicycle")


def test_metrics_count_wrong_answers_as_fp_and_fn():
    df = pd.DataFrame({
        "gold_object": ["knife", "floor", "bicycle", "chair"],
        "extracted_object": ["KNIFE", "OTHER", "car", "chair"],
    })
    m = metrics(score(df), "strict")
    assert (m["TP"], m["FP"], m["FN"]) == (2, 1, 2)
    assert round(m["precision"], 3) == round(2 / 3, 3)
    assert m["recall"] == 0.5


def test_main_end_to_end_with_fake_model(tmp_path):
    def fake_runner(gold, config):
        # Gets every case right except the first, which it misses.
        preds = gold[["case_number"]].copy()
        preds["extracted_object"] = gold["gold_object"].str.split("/").str[0]
        preds.loc[0, "extracted_object"] = "OTHER"
        return preds

    table = evaluate.main(
        ["--details", str(tmp_path / "d.csv"), "--summary", str(tmp_path / "s.md")],
        model_runner=fake_runner,
    )
    overall = table[(table["subset"] == "all") & (table["rule"] == "lenient")].iloc[0]
    assert overall["FP"] == 0 and overall["FN"] == 1
    assert (tmp_path / "s.md").exists()