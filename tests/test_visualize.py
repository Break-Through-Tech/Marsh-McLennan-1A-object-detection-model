import pandas as pd
import pytest

from src.visualize import (
    LONG_TAIL,
    admission_rates,
    categorize,
    category_shares,
    eval_metrics,
    field_matrix,
    make_all,
    real_objects,
    value_totals,
)


def make_ranking():
    return pd.DataFrame(
        {
            "rank": [1, None, 2, 3, 4],
            "object": ["FLOOR", "OTHER", "BED", "KNIFE", "WIDGET"],
            "count": [50, 20, 15, 10, 5],
            "pct": [50.0, 20.0, 15.0, 10.0, 5.0],
            "cum_pct": [50.0, 70.0, 85.0, 95.0, 100.0],
        }
    )


def make_by_field():
    rows = [
        # field, value, object, count, pct_of_object, pct_of_value
        ("age_group", "0-4", "BED", 10, 66.7, 50.0),
        ("age_group", "65+", "BED", 5, 33.3, 10.0),
        ("age_group", "65+", "FLOOR", 40, 80.0, 80.0),
        ("age_group", "0-4", "FLOOR", 10, 20.0, 50.0),
        ("location", "HOME", "BED", 15, 100.0, 30.0),
        ("location", "HOME", "FLOOR", 35, 70.0, 70.0),
        ("location", "STREET", "FLOOR", 15, 30.0, 100.0),
        ("disposition", "1 - TREATED/EXAMINED AND RELEASED", "FLOOR", 30, 60.0, 60.0),
        ("disposition", "4 - TREATED AND ADMITTED/HOSPITALIZED", "FLOOR", 20, 40.0, 80.0),
        ("disposition", "1 - TREATED/EXAMINED AND RELEASED", "BED", 15, 100.0, 30.0),
        ("disposition", "4 - TREATED AND ADMITTED/HOSPITALIZED", "STRIKING HEAD", 5, 100.0, 20.0),
    ]
    return pd.DataFrame(rows, columns=["field", "value", "object", "count", "pct_of_object", "pct_of_value"])


def test_categorize_known_and_long_tail():
    assert categorize("FLOOR") == "Floors and ground"
    assert categorize("knife") == "Tools and sharp objects"
    assert categorize("WIDGET") == LONG_TAIL


def test_real_objects_drops_fallback_and_rescales():
    real = real_objects(make_ranking().assign(category="x"))
    assert "OTHER" not in set(real["object"])
    assert real["pct_of_extracted"].sum() == pytest.approx(100.0)
    assert real["pct_of_extracted"].iloc[0] == pytest.approx(62.5)


def test_category_shares_puts_long_tail_last():
    r = make_ranking()
    r["category"] = r["object"].map(categorize)
    cats = category_shares(r)
    assert cats["category"].iloc[-1] == LONG_TAIL
    assert cats["count"].sum() == 80
    assert cats.loc[cats.category == "Floors and ground", "pct_of_extracted"].iloc[0] == pytest.approx(62.5)


def test_field_matrix_orders_and_fills():
    m = field_matrix(make_by_field(), "age_group", ["FLOOR", "BED"], ["0-4", "5-14", "65+"])
    assert list(m.columns) == ["0-4", "65+"]  # unseen value dropped
    assert m.loc["FLOOR", "65+"] == 80.0


def test_value_totals_recovers_group_sizes():
    t = value_totals(make_by_field(), "location")
    assert t["HOME"] == 50 and t["STREET"] == 15


def test_admission_rates_skips_non_objects():
    rates = admission_rates(make_by_field())
    assert "STRIKING HEAD" not in rates.index
    assert rates.loc["FLOOR", "serious_pct"] == pytest.approx(40.0)
    assert rates.loc["BED", "serious_pct"] == 0


def test_eval_metrics_matches_rules():
    d = pd.DataFrame(
        {
            "extracted_object": ["A", "B", "OTHER", "C"],
            "strict_correct": [True, False, False, False],
            "lenient_correct": [True, True, False, False],
        }
    )
    m = eval_metrics(d)
    assert m["strict"]["precision"] == pytest.approx(1 / 3)
    assert m["strict"]["recall"] == pytest.approx(1 / 4)
    assert m["lenient"]["precision"] == pytest.approx(2 / 3)


def test_make_all_writes_charts_and_report(tmp_path):
    make_ranking().to_csv(tmp_path / "task11_object_ranking.csv", index=False)
    by_field = make_by_field()
    extra = by_field[by_field.field == "location"].assign(field="body_part", value="75 - HEAD")
    extra2 = by_field[by_field.field == "location"].assign(field="diagnosis", value="57 - FRACTURE")
    pd.concat([by_field, extra, extra2]).to_csv(tmp_path / "task11_object_by_field.csv", index=False)
    out = make_all(tmp_path, top=4)
    assert out["report"].exists()
    pngs = list((tmp_path / "figures").glob("task12_*.png"))
    assert len(pngs) == 9  # no eval chart without the Task #7/#9 CSVs
    assert "figures/task12_top_objects.png" in out["report"].read_text()
