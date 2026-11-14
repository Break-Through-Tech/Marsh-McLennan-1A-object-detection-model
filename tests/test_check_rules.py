import pandas as pd

from src import check_rules


def test_rule_agreement_counts_matches_per_rule():
    df = pd.DataFrame({
        "extracted_object": ["FLOOR", "FLOOR", "BED", "KNIFE", "OTHER"],
        "extracted_entity_type": ["rule: floor default", "rule: floor default", "rule: keyword", "tool", None],
        "product_label": ["FLOORS OR FLOORING MATERIALS", "STAIRS OR STEPS",
                          "BEDS OR BEDFRAMES, OTHER OR NOT SPECIFIED", "KNIVES", None],
    })
    t = check_rules.rule_agreement(df)
    floor = t[t["extracted_object"] == "FLOOR"].iloc[0]
    assert (floor["fired"], floor["agree"]) == (2, 1)
    bed = t[t["extracted_object"] == "BED"].iloc[0]
    assert (bed["fired"], bed["agree"]) == (1, 1)
    assert len(t) == 2          # model spans and OTHER rows are not counted


def test_main_prints_summary(tmp_path, capsys):
    path = tmp_path / "x.csv"
    pd.DataFrame({
        "extracted_object": ["FLOOR", "OTHER"], "extracted_entity_type": ["rule: floor default", None],
        "product_label": ["FLOORS OR FLOORING MATERIALS", "X"],
    }).to_csv(path, index=False)
    check_rules.main(["--input", str(path)])
    out = capsys.readouterr().out
    assert "Rule-made predictions: 1" in out and "100.0%" in out
