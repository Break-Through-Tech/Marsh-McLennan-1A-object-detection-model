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


# ---------------------------------------------------------------- Task #9 rules

class StaticModel:
    """Returns the same entities for every text."""

    def __init__(self, entities):
        self.entities = entities

    def predict_entities(self, text, labels, threshold=0.5):
        return list(self.entities)


def ent(text, score=0.6):
    return {"text": text, "label": "object", "score": score, "start": 0, "end": 0}


def run_one(narrative, entities=(), **config):
    df = pd.DataFrame({"case_number": [1], "narrative": [narrative]})
    out = extract.run(df, StaticModel(list(entities)), ExtractionConfig(**config))
    return out.loc[0]


def test_fall_with_no_object_defaults_to_floor():
    row = run_one("70YOF FELL LAST NIGHT AT HOME IN HER KITCHEN. DX ULNA AND RADIUS FX")
    assert row["extracted_object"] == "FLOOR"
    assert row["extracted_entity_type"] == extract.FLOOR_ENTITY_TYPE


def test_floor_mention_without_a_fall_word():
    row = run_one("21YOM HAD SEIZURE WHILE STANDING AND HIT BACK OF HEAD ON TILE FLOOR DX: CHI")
    assert row["extracted_object"] == "FLOOR"


def test_dropped_place_span_then_floor_default():
    text = "81YOF PT C/O DIZZINESS CAUSING A FALL ON WAY TO BATHROOM, FOUND ON THE FLOOR DX: COVID 19"
    assert run_one(text, [ent("BATHROOM")])["extracted_object"] == "FLOOR"


def test_dropped_vehicle_exposes_next_candidate():
    text = "28YOM PT WAS AN UNHELMETED BICYCLE RIDER, STRUCK BY A CAR WHILE RIDING BIKE DX; CONTUSION"
    row = run_one(text, [ent("CAR", 0.9), ent("BICYCLE", 0.4)])
    assert row["extracted_object"] == "BICYCLE"


def test_stoplist_matches_the_whole_span_only():
    row = run_one("2YOM FELL OUT OF A CAR SEAT DX: CONTUSION", [ent("car seat")])
    assert row["extracted_object"] == "car seat"


def test_no_floor_default_for_outdoor_falls_or_crashes():
    assert run_one("44YOM FELL ON THE SIDEWALK DX: SPRAIN")["extracted_object"] == "OTHER"
    pond = "11YOF PRESENTS AFTER A FALL WHILE ICE SKATING ON A POND DX: TORUS FRACTURE"
    assert run_one(pond, [ent("POND")])["extracted_object"] == "OTHER"
    crash = "33YOM ON MOPED WAS HIT BY A CAR IN AN INTERSECTION AND FELL ONTO RIGHT SIDE DX: KNEE PAIN"
    assert run_one(crash, [ent("INTERSECTION")])["extracted_object"] == "OTHER"


def test_no_floor_default_when_text_is_redacted_or_fell_asleep():
    assert run_one("2YOM WAS WALKING IN SIBLINGS *** AND SUSTAINED FALL DX: LIMP")["extracted_object"] == "OTHER"
    assert run_one("SHE FELL ASLEEP HOLDING THE BABY DX: NONE")["extracted_object"] == "OTHER"


def test_a_model_span_beats_the_floor_default():
    row = run_one("70YOM FELL DOWN STEPS. DX: FX", [ent("STEPS")])
    assert row["extracted_object"] == "STEPS"


def test_floor_default_can_be_switched_off():
    row = run_one("70YOF FELL AT HOME. DX: FX", fallback_rule=None)
    assert row["extracted_object"] == "OTHER"


def test_summarize_run_counts_fallback_and_failures():
    result = pd.DataFrame(
        {
            "case_number": [1, 2, 3, 4],
            "extracted_object": ["LADDER", "OTHER", None, "LADDER"],
            "extracted_entity_type": ["object", None, None, "object"],
        }
    )
    stats = extract.summarize_run(result, 4, 2.0, ExtractionConfig())
    assert stats["input_rows"] == 4 and stats["output_rows"] == 4
    assert stats["runtime_seconds"] == 2.0 and stats["rows_per_second"] == 2.0
    assert stats["fallback_rows"] == 1 and stats["fallback_pct"] == 25.0
    assert stats["failed_rows"] == 1 and stats["failed_case_numbers"] == [3]
    assert stats["top_objects"] == {"LADDER": 2, "OTHER": 1}


def test_main_writes_run_log(tmp_path, monkeypatch):
    import json

    inp = tmp_path / "in.csv"
    sample_df().to_csv(inp, index=False)
    monkeypatch.setattr(extract, "load_model", lambda config: FakeModel())
    out, log = tmp_path / "out.csv", tmp_path / "log.json"
    extract.main(["--input", str(inp), "--output", str(out), "--log", str(log)])
    stats = json.loads(log.read_text(encoding="utf-8"))
    assert stats["input_rows"] == stats["output_rows"] == len(pd.read_csv(out)) == 4
    assert stats["failed_rows"] == 0


# ---------------------------------------------------------------- Task #2 span cleaning

def test_clean_span_drops_action_words_and_keeps_products():
    from src.extract import default_clean_span as clean
    assert clean("HANDLING KNIFE") == "KNIFE"
    assert clean("BASKETBALL GAME") == "BASKETBALL"
    assert clean("HIS BIKE") == "BIKE"
    assert clean("VIDEO GAME") == "VIDEO GAME"          # a real product, protected
    assert clean("RIDING MOWER") == "RIDING MOWER"      # a real product
    assert clean("LADDER") == "LADDER"
    assert clean("THE VIDEO GAME") == "VIDEO GAME"
    assert clean("BOARD GAME ACCIDENT") == "BOARD GAME"


def test_clean_span_fixes_near_miss_spellings_only():
    from src.extract import default_clean_span as clean
    assert clean("WHELLCHAIR") == "WHEELCHAIR"
    assert clean("STAIRWAY") == "STAIRWAY"              # not close to any vocabulary word
    assert clean("wheelchair") == "wheelchair"


def test_cleaned_span_reaches_the_stoplist():
    row = run_one("21YOF HIT HER HEAD ON THE SHELF DX: CHI", [ent("HER HEAD")])
    assert row["extracted_object"] != "HER HEAD"        # cleaned to HEAD, then dropped as a body part


def test_clean_span_can_be_switched_off():
    row = run_one("37YOM HANDLING KNIFE, CUT LEG DX: LAC", [ent("HANDLING KNIFE")], clean_span=None)
    assert row["extracted_object"] == "HANDLING KNIFE"


# ---------------------------------------------------------------- Task #2 keyword rule

def test_keyword_rule_names_a_missed_household_object():
    row = run_one("74 YOF. HEAD INJURY AFTER PATIENT FELL OUT OF BED AT THE NURSING HOME. DX: SDH")
    assert row["extracted_object"] == "BED"
    assert row["extracted_entity_type"] == extract.KEYWORD_ENTITY_TYPE
    assert run_one("3YOM FALL DOWN 5 STEPS DX CHEST WALL PAIN")["extracted_object"] == "STAIRS"
    assert run_one("9YOF HAD A SLIP AND FALL IN THE SHOWER STRUCK HEAD DX: CHI")["extracted_object"] == "SHOWER"


def test_keyword_rule_beats_the_floor_default_and_earliest_word_wins():
    text = "69YOM FELL OUT OF BED AT CARE FACILITY FOUND ON THE FLOOR, LEFT ELBOW SKIN TEAR"
    assert run_one(text)["extracted_object"] == "BED"
    assert run_one("54YOM FELL DOWN STEPS ONTO A TILE FLOOR DX: FX")["extracted_object"] == "STAIRS"


def test_keyword_rule_skips_dirt_bikes_and_whole_word_misses():
    assert run_one("16YOM DIRT BIKE ACCIDENT DX: CONCUSSION")["extracted_object"] == "OTHER"
    assert run_one("5YOM IN THE BATHROOM WITH A BEDROOM DOOR DX: LAC")["extracted_object"] == "OTHER"


def test_a_model_span_beats_the_keyword_rule():
    row = run_one("70YOM FELL OUT OF BED ONTO HIS CANE DX: FX", [ent("CANE")])
    assert row["extracted_object"] == "CANE"


def test_side_words_are_stripped_so_body_parts_reach_the_stoplist():
    from src.extract import default_clean_span as clean
    assert clean("RIGHT KNEE") == "KNEE"
    assert clean("LOWER BACK") == "BACK"
    text = "20YOM FELL RUNNING ON CONCRETE FLOOR DX: ABRASION"
    assert run_one(text, [ent("RIGHT KNEE")])["extracted_object"] == "FLOOR"


def test_keyword_and_floor_rules_switch_off_together():
    assert run_one("74 YOF FELL OUT OF BED DX: SDH", fallback_rule=None)["extracted_object"] == "OTHER"


# ---------------------------------------------------------------- v4 speed: batching, shards, resume

class InferenceModel:
    """Records inference() calls; finds LADDER and BED."""

    def __init__(self):
        self.calls = []

    def inference(self, texts, labels, threshold=0.5, batch_size=8):
        self.calls.append((list(texts), batch_size))
        out = []
        for t in texts:
            found = [w for w in ("LADDER", "BED") if w in t]
            out.append([{"text": w, "label": "object", "score": 0.9, "start": 0, "end": 0} for w in found])
        return out


def test_predict_entities_dedupes_sorts_and_keeps_input_order():
    model = InferenceModel()
    texts = ["FELL OFF A LONG LADDER", "BED", "FELL OFF A LONG LADDER", "NOTHING HERE"]
    out = extract.predict_entities(model, texts, ExtractionConfig(batch_size=24))
    assert [[e["text"] for e in ents] for ents in out] == [["LADDER"], ["BED"], ["LADDER"], []]
    sent = [t for call, _ in model.calls for t in call]
    assert len(sent) == 3                                   # the duplicate ran once
    assert sent == sorted(sent, key=len)                    # shortest first
    assert all(bs == 24 for _, bs in model.calls)           # the real batch size reaches GLiNER


def test_select_shard_splits_rows_without_overlap():
    df = pd.DataFrame({"case_number": range(10)})
    parts = [extract.select_shard(df, f"{k}/3") for k in (1, 2, 3)]
    ids = [i for p in parts for i in p["case_number"]]
    assert sorted(ids) == list(range(10)) and len(ids) == 10


def test_chunked_run_resumes_and_merge_combines(tmp_path, monkeypatch):
    clean = tmp_path / "clean.csv"
    pd.DataFrame({
        "case_number": [1, 2, 3, 4, 5],
        "narrative": ["FELL OFF LADDER DX: FX", "FELL OUT OF BED DX: FX", "CUT FINGER DX: LAC",
                      "LADDER TIPPED DX: FX", "BED FRAME DX: CONT"],
    }).to_csv(clean, index=False)
    model = InferenceModel()
    monkeypatch.setattr(extract, "load_model", lambda config: model)

    out = tmp_path / "out.csv"
    for shard in ("1/2", "2/2"):
        extract.main(["--input", str(clean), "--output", str(out), "--shard", shard, "--chunk-size", "2"])
    shard1 = tmp_path / "out.shard1of2.csv"
    assert len(list((tmp_path / "out.shard1of2_parts").glob("part_*.csv"))) == 2

    calls_before = len(model.calls)
    extract.main(["--input", str(clean), "--output", str(out), "--shard", "1/2", "--chunk-size", "2"])
    assert len(model.calls) == calls_before                # every chunk was reused, model not called

    final = tmp_path / "final.csv"
    extract.main(["--merge", str(shard1), str(tmp_path / "out.shard2of2.csv"), "--output", str(final)])
    merged = pd.read_csv(final)
    assert sorted(merged["case_number"]) == [1, 2, 3, 4, 5]
    assert merged.set_index("case_number").loc[1, "extracted_object"] == "LADDER"
