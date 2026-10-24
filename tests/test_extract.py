"""Tests for src/extract.py. A fake model stands in for GLiNER, so these run
without downloading weights:  python -m pytest tests"""

import pandas as pd

from src import extract
from src.extract import ExtractionConfig


class FakeModel:
    """Returns an entity for every known object word found in the text it is given."""

    OBJECTS = {"LADDER": 0.9, "KNIFE": 0.8, "HAND": 0.95, "BED": 0.5, "FOREIGN BODY": 0.99}

    def __init__(self):
        self.seen_texts = []
        self.seen_labels = []

    def predict_entities(self, text, labels, threshold=0.5):
        self.seen_texts.append(text)
        self.seen_labels.append(list(labels))
        found = []
        for word, score in self.OBJECTS.items():
            start = text.upper().find(word)
            if start >= 0 and score >= threshold:
                found.append(
                    {"text": text[start : start + len(word)], "label": labels[0],
                     "score": score, "start": start, "end": start + len(word)}
                )
        return found


def sample_df():
    return pd.DataFrame(
        {
            "case_number": [1, 2, 3, 4],
            "narrative": [
                "47YOF FELL OFF A LADDER AND HURT HER HAND. DX: FRACTURE",
                "30 YOM CUT HIMSELF W/KNIFE. DX FOREIGN BODY IN FINGER",
                "5YOM ROLLED OFF BED DX: CONTUSION",
                "20YOF WALKING ON SIDEWALK. DX: SPRAIN",
            ],
            "product_code": [1, 2, 3, 4],
        }
    )


def test_one_row_out_per_row_in():
    out = extract.run(sample_df(), FakeModel())
    assert len(out) == 4
    assert out["case_number"].tolist() == [1, 2, 3, 4]


def test_event_text_excludes_diagnosis_section():
    model = FakeModel()
    out = extract.run(sample_df(), model)
    assert out.loc[1, "extracted_object"] == "KNIFE"  # not FOREIGN BODY from the DX section
    assert all("DX" not in t for t in model.seen_texts)


def test_normalized_text_field_keeps_diagnosis():
    out = extract.run(sample_df(), FakeModel(), ExtractionConfig(text_field="normalized"))
    assert out.loc[1, "extracted_object"] == "FOREIGN BODY"


def test_fallback_when_nothing_found():
    out = extract.run(sample_df(), FakeModel())
    assert out.loc[3, "extracted_object"] == "OTHER"
    assert pd.isna(out.loc[3, "extraction_confidence"])


def test_blank_narrative_gets_fallback_without_calling_model():
    df = sample_df().assign(narrative=["", None, "   ", "5YOM ROLLED OFF BED DX: CONTUSION"])
    model = FakeModel()
    out = extract.run(df, model)
    assert out["extracted_object"].tolist()[:3] == ["OTHER"] * 3
    assert len(model.seen_texts) == 1


def test_default_rule_drops_body_part():
    out = extract.run(sample_df(), FakeModel())
    # HAND scores higher than LADDER but is a body part, so LADDER wins
    assert out.loc[0, "extracted_object"] == "LADDER"


def test_swap_rules():
    keep_everything = ExtractionConfig(keep_entity=lambda e: True)
    out = extract.run(sample_df(), FakeModel(), keep_everything)
    assert out.loc[0, "extracted_object"] == "HAND"

    first_span = ExtractionConfig(keep_entity=lambda e: True, choose_entity=lambda es: es[0])
    out = extract.run(sample_df(), FakeModel(), first_span)
    assert out.loc[0, "extracted_object"] == "LADDER"


def test_swap_labels_and_threshold():
    model = FakeModel()
    config = ExtractionConfig(labels=("product",), threshold=0.85)
    out = extract.run(sample_df(), model, config)
    assert set(map(tuple, model.seen_labels)) == {("product",)}
    assert out.loc[1, "extracted_object"] == "OTHER"  # KNIFE scores 0.8, below 0.85


def test_lowercase_option_changes_model_input_only():
    model = FakeModel()
    out = extract.run(sample_df(), model, ExtractionConfig(lowercase=True))
    assert all(t == t.lower() for t in model.seen_texts)
    assert out.loc[0, "narrative_normalized"].isupper()


def test_passthrough_columns_and_cli(tmp_path, monkeypatch):
    csv_in = tmp_path / "clean.csv"
    csv_out = tmp_path / "out.csv"
    sample_df().to_csv(csv_in, index=False)
    monkeypatch.setattr(extract, "load_model", lambda config: FakeModel())

    extract.main(["--input", str(csv_in), "--output", str(csv_out), "--sample", "3"])

    result = pd.read_csv(csv_out)
    assert len(result) == 3
    assert {"case_number", "event_text", "extracted_object", "product_code"} <= set(result.columns)
