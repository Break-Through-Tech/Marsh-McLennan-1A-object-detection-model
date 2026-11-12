import pandas as pd
import pytest

from src.summarize import (
    age_group,
    build_frame,
    build_report,
    coverage,
    decode,
    load_codebook,
    objects_by_field,
    rank_objects,
    top_objects_per_value,
)

FMT = pd.DataFrame(
    {
        "Format name": ["BDYPT", "BDYPT", "LOC", "LOC"],
        "Starting value for format": [30, 75, 0, 1],
        "Ending value for format": [30, 75, 0, 1],
        "Format value label": ["SHOULDER", "HEAD", "UNK", "HOME"],
    }
)


def make_inputs():
    ext = pd.DataFrame(
        {
            "case_number": [1, 2, 3, 4, 5, 6],
            "extracted_object": ["FLOOR", "BED", "BED", "OTHER", "KNIFE", "BED"],
        }
    )
    clean = pd.DataFrame(
        {
            "case_number": [1, 2, 3, 4, 5, 6],
            "age_years": [3, 70, 70, 30, 30, None],
            "body_part": [75, 75, 30, 30, 30, 75],
            "location": [1, 1, 1, None, 1, 1],
        }
    )
    return ext, clean


def test_age_group_buckets_and_missing():
    out = age_group(pd.Series([0, 4, 5, 65, 90, None]))
    assert out.tolist()[:5] == ["0-4", "0-4", "5-14", "65+", "65+"]
    assert pd.isna(out.iloc[5])


def test_decode_uses_codebook_ranges():
    table = load_codebook(FMT)["BDYPT"]
    out = decode(pd.Series([75, 30, 99, None]), table)
    assert out.tolist()[:2] == ["HEAD", "SHOULDER"]
    assert pd.isna(out.iloc[2]) and pd.isna(out.iloc[3])


def test_build_frame_keeps_row_count_and_decodes():
    ext, clean = make_inputs()
    frame = build_frame(ext, clean, FMT)
    assert len(frame) == 6
    assert frame.loc[frame.case_number == 2, "body_part"].iloc[0] == "HEAD"
    assert frame.loc[frame.case_number == 6, "age_group"].isna().all()


def test_build_frame_requires_columns():
    ext, clean = make_inputs()
    with pytest.raises(KeyError):
        build_frame(ext.drop(columns="extracted_object"), clean, FMT)


def test_rank_objects_skips_fallback_in_rank():
    ext, clean = make_inputs()
    r = rank_objects(build_frame(ext, clean, FMT))
    assert r["count"].sum() == 6
    assert r.iloc[0]["object"] == "BED" and r.iloc[0]["rank"] == 1
    assert pd.isna(r.loc[r["object"] == "OTHER", "rank"].iloc[0])
    assert r["cum_pct"].iloc[-1] == pytest.approx(100.0)


def test_objects_by_field_shares():
    ext, clean = make_inputs()
    frame = build_frame(ext, clean, FMT)
    t = objects_by_field(frame, fields=["body_part"])
    assert "OTHER" not in set(t["object"])
    row = t[(t.object == "BED") & (t.value == "HEAD")].iloc[0]
    assert row["count"] == 2
    assert row["pct_of_object"] == pytest.approx(100 * 2 / 3)
    assert row["pct_of_value"] == pytest.approx(100 * 2 / 3)  # HEAD cases excl. OTHER: FLOOR, BED, BED


def test_unknown_group_for_missing_values():
    ext, clean = make_inputs()
    frame = build_frame(ext, clean, FMT)
    t = top_objects_per_value(frame, "age_group", n=3, min_cases=1)
    assert "Unknown" in set(t["value"])


def test_coverage_and_report_render():
    ext, clean = make_inputs()
    frame = build_frame(ext, clean, FMT)
    cov = coverage(frame)
    assert cov["rows"] == 6 and cov["fallback_rows"] == 1 and cov["distinct_objects"] == 3
    report = build_report(frame, rank_objects(frame), objects_by_field(frame), top=5)
    assert "Top 5 objects" in report and "| BED |" in report
