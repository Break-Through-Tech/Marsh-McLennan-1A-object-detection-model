"""Tests for src/final_eval.py. No model download needed:  python -m pytest tests"""

import numpy as np
import pandas as pd
import pytest

from src import final_eval as fe


def test_micro_counts_wrong_answers_as_fp_and_fn():
    correct = np.array([True, False, False, True])
    answered = np.array([True, True, False, True])
    m = fe.micro(correct, answered)
    assert (m["TP"], m["FP"], m["FN"]) == (2, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(0.5)


def test_macro_matches_hand_computation():
    gold = np.array([1, 1, 2, 3])
    pred = np.array([1, 2, 2, fe.ABSTAIN])
    m = fe.macro(gold, pred)
    # class 1: P 1/1 R 1/2 F1 2/3; class 2: P 1/2 R 1/1 F1 2/3; class 3: P 0 R 0 F1 0
    assert m["classes"] == 3
    assert m["precision"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert m["recall"] == pytest.approx((0.5 + 1 + 0) / 3)
    assert m["f1"] == pytest.approx((2 / 3 + 2 / 3) / 3)


def test_macro_restricted_classes_ignore_absent_ones():
    gold = np.array([1, 1, 2])
    pred = np.array([1, 1, 1])
    assert fe.macro(gold, pred, classes=np.array([1, 99]))["classes"] == 1


def test_mapper_prefers_head_word_and_pairs():
    texts = pd.Series(["FELL ON FLOOR"] * 30 + ["CUT WITH KITCHEN KNIFE"] * 30 + ["KITCHEN FLOOR WET"] * 25)
    codes = pd.Series([1807] * 30 + [464] * 30 + [1807] * 25)
    mapper = fe.CodeMapper.fit(texts, codes, min_support=20)
    assert mapper.map_span("KNIFE") == 464
    assert mapper.map_span("knives") == 464  # plural trimmed like src/evaluate.py
    assert mapper.map_span("KITCHEN KNIFE") == 464  # the pair wins over the ambiguous word
    assert mapper.map_span("CONCRETE FLOOR") == 1807  # head word
    assert mapper.map_span("SPACESHIP") == fe.UNMAPPED
    assert mapper.map_span(None) == fe.UNMAPPED


def test_keyword_baseline_picks_purest_term():
    texts = pd.Series(["FELL ON FLOOR"] * 30 + ["CUT WITH KNIFE ON FLOOR"] * 30)
    codes = pd.Series([1807] * 30 + [464] * 30)
    mapper = fe.CodeMapper.fit(texts, codes, min_support=20)
    span, code = mapper.keyword_predict("PT CUT FINGER WITH KNIFE")
    assert code == 464 and "KNIFE" in span
    assert mapper.keyword_predict("NOTHING HERE")[1] == fe.ABSTAIN


def _clean():
    return pd.DataFrame({
        "case_number": ["1", "2", "3", "4", "5", "6"],
        "narrative": ["A", "B", "SHARED", "SHARED", "C", "D"],
        "product_code": [1807, 464, 1807, 1807, 4076, 1842],
        "product_label": ["FLOORS", "KNIVES", "FLOORS", "FLOORS", "BEDS", "STAIRS"],
        "product_2_code": [0] * 6, "product_3_code": [0] * 6,
        "vague_label": [False] * 6, "rare_class": [False] * 6, "redacted": [False] * 6,
        "psu": [1, 1, 1, 2, 1, 1],
    })


def test_build_test_set_excludes_dev_cases_and_cross_fold_duplicates():
    folds = {1: "test", 2: "train"}
    test = fe.build_test_set(_clean(), folds, exclude={"1"}, n=10, seed=0)
    assert set(test["case_number"]) == {"2", "5", "6"}  # 1 excluded, 3 shares text with train case 4


def test_gold_code_set_drops_zero_slots():
    row = {"product_code": 1807, "product_2_code": 0, "product_3_code": 4076}
    assert fe.gold_code_set(row) == frozenset({1807, 4076})


def test_error_category_order():
    base = {"v3_correct": False, "v3_code": 464, "v3_set_match": False,
            "v3_entity_type": "object", "vague_label": False}
    cat = lambda row: fe.error_category(row, name="v3")  # noqa: E731
    assert cat({**base, "v3_correct": True}) == "correct"
    assert cat({**base, "v3_code": fe.ABSTAIN}) == "abstained (OTHER)"
    assert cat({**base, "v3_code": fe.UNMAPPED}) == "span has no lexicon code"
    assert cat({**base, "v3_set_match": True}) == "matches a secondary NEISS product"
    assert cat({**base, "v3_entity_type": "rule: floor default"}) == "floor/keyword rule wrong"
    assert cat(base) == "model span, different product"
    clf = {"classifier_correct": False, "classifier_code": 464, "classifier_set_match": False,
           "classifier_entity_type": "classifier"}
    assert fe.error_category(clf) == "different product"  # FINAL is the classifier


def test_comparisons_put_final_first_then_history():
    pairs = fe.comparisons(["v1", "v2", "v3", "v4", "classifier"])
    assert pairs[:4] == [("classifier", "v1"), ("classifier", "v2"), ("classifier", "v3"), ("classifier", "v4")]
    assert pairs[4:] == [("v3", "v1"), ("v3", "v2"), ("v4", "v3")]


def test_label_in_text_heuristic():
    assert fe.label_in_text("47YOF FELL DOWN STAIRS", "STAIRS OR STEPS")
    assert not fe.label_in_text("47YOF FELL", "STAIRS OR STEPS")


class FakeGliner:
    """Returns the last word of the text as an 'object' span, except for texts with NOTHING."""

    def predict_entities(self, text, labels, threshold=0.5):
        if "NOTHING" in text:
            return []
        word = text.split()[-1]
        return [{"start": 0, "end": len(word), "text": word, "label": "object", "score": 0.9}]


def test_main_end_to_end_with_fake_model(tmp_path):
    rng = np.random.default_rng(0)
    objects = [("KNIFE", 464, "KNIVES"), ("BED", 4076, "BEDS OR BEDFRAMES"), ("LADDER", 620, "LADDERS")]
    rows = []
    for i in range(400):
        obj, code, label = objects[i % 3]
        psu = 1 if i < 300 else 2
        rows.append({"case_number": str(1000 + i), "narrative": f"{i}YOM HURT BY {obj} DX: X",
                     "event_text": f"{i}YOM HURT BY {obj}", "product_code": code, "product_label": label,
                     "product_2_code": 0, "product_3_code": 0, "vague_label": False, "rare_class": False,
                     "redacted": False, "psu": psu})
    rows.append({**rows[-1], "case_number": "9999", "narrative": "1YOM NOTHING NAMED DX: X",
                 "event_text": "1YOM NOTHING NAMED"})
    clean = tmp_path / "clean.csv"
    pd.DataFrame(rows).to_csv(clean, index=False)
    pd.DataFrame({"PSU": [1, 2], "fold": ["train", "test"]}).to_csv(tmp_path / "split.csv", index=False)
    pd.DataFrame({"format": ["PROD"] * 3, "code": [464, 4076, 620], "end": [464, 4076, 620],
                  "label": ["464 - KNIVES", "4076 - BEDS OR BEDFRAMES", "620 - LADDERS"]}
                 ).to_csv(tmp_path / "fmt.csv", index=False)
    pd.DataFrame({"case_number": ["1000"], "narrative": ["0YOM HURT BY KNIFE DX: X"],
                  "gold_object": ["knife"], "object_code": ["464"], "mention_type": ["explicit"]}
                 ).to_csv(tmp_path / "dev.csv", index=False)

    out = lambda name: str(tmp_path / name)  # noqa: E731
    ctx = fe.main(["--clean", str(clean), "--split", out("split.csv"), "--fmt", out("fmt.csv"),
                   "--dev-gold", out("dev.csv"), "--test-set", out("test.csv"),
                   "--hand-labels", out("hand.csv"), "--cache-dir", out("cache"),
                   "--details", out("details.csv"), "--metrics", out("metrics.csv"),
                   "--summary", out("summary.md"), "--error-sample-dir", out("errors"), "--figure", out("fig.png"),
                   "--n", "50", "--bootstrap", "20", "--make-label-sheet"],
                  model_loader=lambda config: FakeGliner())

    m = ctx["metrics"].set_index("system")
    assert m.loc["v3", "micro_f1"] == pytest.approx(1.0)  # fake model names every object
    assert m.loc["v3", "macro_f1"] > m.loc["majority", "macro_f1"]
    assert (tmp_path / "hand.csv").exists() and (tmp_path / "summary.md").exists()
    assert {p.name for p in (tmp_path / "errors").iterdir()} >= {"final_eval_error_sample_v3.csv"}
    summary = (tmp_path / "summary.md").read_text()
    assert "## v3 error categories" in summary and "| v3 - v1 | micro_f1 |" in summary
    test = pd.read_csv(tmp_path / "test.csv", dtype={"case_number": str})
    assert set(test["psu"]) == {2} and "1000" not in set(test["case_number"])
    # A second run reads the cached entities and never loads the model.
    fe.main(["--clean", str(clean), "--split", out("split.csv"), "--fmt", out("fmt.csv"),
             "--dev-gold", out("dev.csv"), "--test-set", out("test.csv"),
             "--hand-labels", out("hand.csv"), "--cache-dir", out("cache"),
             "--details", out("details.csv"), "--metrics", out("metrics.csv"),
             "--summary", out("summary.md"), "--error-sample-dir", out("errors"), "--figure", out("fig.png"),
             "--n", "50", "--bootstrap", "20"],
            model_loader=lambda config: pytest.fail("model should not load on a cached rerun"))
