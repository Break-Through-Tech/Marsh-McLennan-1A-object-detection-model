"""Tests for src/tune.py with a fake model, so no weights are needed:  python -m pytest tests"""

import pandas as pd

from src import extract, tune


class FakeModel:
    """Finds KNIFE (wrapped in an action word) always, and FOOTBALL only when asked for a sport."""

    def predict_entities(self, text, labels, threshold=0.5):
        found = []
        if "KNIFE" in text:
            found.append({"text": "HANDLING KNIFE", "label": "tool", "score": 0.6, "start": 0, "end": 0})
        if "FOOTBALL" in text and "sport" in labels:
            found.append({"text": "FOOTBALL", "label": "sport", "score": 0.5, "start": 0, "end": 0})
        return found


def gold_df():
    return pd.DataFrame({
        "case_number": ["1", "2", "3"],
        "narrative": [
            "37YOM HANDLING KNIFE, ACCIDENTALLY CUT RT UPPER LEG. DX: LAC THIGH",
            "14YOM WAS SLAMMED WHILE PLAYING FOOTBALL. DX: CONCUSSION",
            "20YOF WALKING ON SIDEWALK. DX: SPRAIN",
        ],
        "gold_object": ["knife", "football (activity)", "sidewalk"],
        "mention_type": ["explicit", "explicit", "explicit"],
    })


def test_grid_covers_every_configuration():
    grid, scored = tune.run_grid(FakeModel(), gold_df())
    expected = (len(tune.LABEL_SETS) * len(tune.TEXT_VARIANTS) * len(tune.THRESHOLDS)
                * len(tune.CLEAN_OPTIONS) * len(tune.FALLBACK_OPTIONS))
    assert len(grid) == expected == len(scored)


def test_baseline_is_the_task9_setup():
    grid, scored = tune.run_grid(FakeModel(), gold_df())
    base = scored[tune.baseline_key()]
    # no span cleaning: HANDLING KNIFE is only a partial match; football is missed
    assert base.loc[base.case_number == "1", "result"].iloc[0] == "partial match"
    assert base.loc[base.case_number == "2", "result"].iloc[0] == "no prediction (OTHER)"


def test_best_configuration_turns_on_cleaning_and_sport_label():
    grid, scored = tune.run_grid(FakeModel(), gold_df())
    best = tune.rank(grid).iloc[0]
    assert best["clean_span"] == "on"
    assert best["labels"] != "base"
    changes = tune.case_changes(scored[tune.baseline_key()], scored[tune.key_of(best)])
    assert set(changes["case_number"]) == {"1", "2"}
    assert set(changes["change"]) == {"fixed"}


def test_main_writes_both_files(tmp_path, monkeypatch):
    gold_path = tmp_path / "gold.csv"
    gold_df().to_csv(gold_path, index=False)
    grid_out, details_out = tmp_path / "grid.csv", tmp_path / "details.csv"
    ranked = tune.main(
        ["--ground-truth", str(gold_path), "--grid-out", str(grid_out), "--details-out", str(details_out)],
        model=FakeModel(),
    )
    assert grid_out.exists() and details_out.exists()
    assert len(pd.read_csv(details_out)) == 3
    assert ranked.iloc[0]["strict_f1"] >= ranked.iloc[-1]["strict_f1"]
