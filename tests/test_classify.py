"""Tests for src/classify.py on a tiny synthetic training set."""
import pandas as pd

from src import classify


def toy():
    rows = []
    for _ in range(30):
        rows += [("FELL OFF A LADDER WHILE PAINTING", 4078), ("CUT FINGER WITH KITCHEN KNIFE", 464),
                 ("FELL OUT OF BED AT NIGHT", 4076), ("TRIPPED ON THE STAIRS AT HOME", 1842)]
    df = pd.DataFrame(rows, columns=["event_text", "product_code"])
    return df


def test_fit_predict_and_margin():
    df = toy()
    clf = classify.ProductClassifier.fit(df["event_text"], df["product_code"], min_df=1)
    codes, margins = clf.predict(["SLIPPED ON A LADDER", "BED FRAME", "STAIRS"])
    assert list(codes) == [4078, 4076, 1842]
    assert (margins > 0).all()


def test_add_predictions_flags_agreement(tmp_path):
    df = toy()
    clf = classify.ProductClassifier.fit(df["event_text"], df["product_code"], min_df=1)
    path = tmp_path / "m.pkl"
    clf.save(path)
    clf = classify.ProductClassifier.load(path)
    ex = pd.DataFrame({"narrative": ["45YOM FELL OFF A LADDER DX: FX", "3YOF FELL OUT OF BED DX: CHI"],
                       "event_text": ["45YOM FELL OFF A LADDER", "3YOF FELL OUT OF BED"],
                       "extracted_object": ["LADDER", "OTHER"]})
    out = classify.add_predictions(ex, clf, {4078: "LADDERS", 4076: "BEDS"})
    assert out["predicted_product"].tolist() == ["LADDERS", "BEDS"]
    assert out["agree"].tolist() == [True, False]
