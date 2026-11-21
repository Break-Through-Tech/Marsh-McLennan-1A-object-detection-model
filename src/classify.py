"""v4 product classifier (Milestone 3).

Predicts the NEISS product code of a narrative with a linear model over TF-IDF
word and word-pair features, trained on the training-fold hospitals only.
GLiNER (src/extract.py) still names the object in the text; this model gives
the product category. On 5,000 validation-fold narratives it scored micro F1
0.898 and macro F1 0.737, against 0.853 / 0.592 for the keyword lookup and
0.772 / 0.528 for GLiNER with the v4 rules (src/validate_v4.py).

Run from the repo root (needs data/processed/neiss2025_clean.csv):

    python -m src.classify --train                     # about 5 minutes on a laptop
    python -m src.classify --input data/processed/extractions_v4.csv \\
                           --output data/processed/extractions_v4_coded.csv

The second command adds predicted_code, predicted_product, classifier_margin,
gliner_code and agree (whether GLiNER's object maps to the same code) to a
finished extraction file.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:  # python -m src.classify
    from src import extract, final_eval
except ImportError:  # python src/classify.py
    import extract
    import final_eval

DEFAULT_MODEL_PATH = Path("data/processed/models/product_classifier.pkl")


class ProductClassifier:
    """TF-IDF (words and word pairs) + linear SVM trained with SGD."""

    def __init__(self, vectorizer, model, mapper: Optional[final_eval.CodeMapper] = None):
        self.vectorizer = vectorizer
        self.model = model
        self.mapper = mapper  # span -> code lexicon from the same training rows

    @classmethod
    def fit(cls, texts: pd.Series, codes: pd.Series, seed: int = 0,
            min_df: int = 3, alpha: float = 2e-6, epochs: int = 8) -> "ProductClassifier":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier

        texts = texts.fillna("").astype(str)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True,
                                     max_features=300_000)
        X = vectorizer.fit_transform(texts)
        model = SGDClassifier(loss="hinge", alpha=alpha, max_iter=epochs, tol=None,
                              n_jobs=-1, random_state=seed)
        model.fit(X, codes.astype(int).values)
        mapper = final_eval.CodeMapper.fit(texts, codes, min_support=min(final_eval.LEXICON_MIN_SUPPORT, len(texts)))
        return cls(vectorizer, model, mapper)

    def predict(self, texts) -> Tuple[np.ndarray, np.ndarray]:
        """Predicted codes and margins (top score minus runner-up; higher = surer)."""
        X = self.vectorizer.transform(pd.Series(list(texts)).fillna("").astype(str))
        scores = self.model.decision_function(X)
        if scores.ndim == 1:  # two classes
            scores = np.column_stack([-scores, scores])
        top = scores.argmax(axis=1)
        ordered = np.sort(scores, axis=1)
        return self.model.classes_[top].astype(int), ordered[:, -1] - ordered[:, -2]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "ProductClassifier":
        if not Path(path).exists():
            raise SystemExit(f"{path} not found. Train it first: python -m src.classify --train")
        with open(path, "rb") as f:
            return pickle.load(f)


def model_texts(df: pd.DataFrame) -> pd.Series:
    """The event text (narrative without the DX section), as GLiNER sees it."""
    if "event_text" in df.columns:
        return df["event_text"].fillna("")
    return extract.prepare_text(df["narrative"], extract.ExtractionConfig())["event_text"]


def add_predictions(df: pd.DataFrame, clf: ProductClassifier, names: dict) -> pd.DataFrame:
    """Add the classifier's code and product name, and whether GLiNER's object agrees."""
    out = df.copy()
    codes, margins = clf.predict(model_texts(out))
    out["predicted_code"] = codes
    out["predicted_product"] = [names.get(int(c), str(c)) for c in codes]
    out["classifier_margin"] = np.round(margins, 4)
    if "extracted_object" in out.columns and clf.mapper is not None:
        span = out["extracted_object"].fillna(extract.FALLBACK).astype(str)
        out["gliner_code"] = [final_eval.ABSTAIN if s.upper() == extract.FALLBACK else clf.mapper.map_span(s)
                              for s in span]
        out["agree"] = out["gliner_code"] == out["predicted_code"]
    return out


def train_from_clean(clean_path: Path, split_path: Path = final_eval.SPLIT_PATH) -> ProductClassifier:
    clean = extract.load_clean(clean_path)
    folds = final_eval.load_folds(split_path)
    train = clean[clean["psu"].astype(int).map(folds) == "train"]
    print(f"Training on {len(train):,} training-fold narratives ...")
    return ProductClassifier.fit(train["event_text"], train["product_code"])


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v4 NEISS product classifier")
    p.add_argument("--train", action="store_true", help="train on the training-fold hospitals and save")
    p.add_argument("--clean", type=Path, default=extract.DEFAULT_INPUT)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--input", type=Path, default=None, help="extraction CSV to add predicted codes to")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--fmt", type=Path, default=final_eval.FMT_PATH, help="NEISS codebook CSV for product names")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.train:
        clf = train_from_clean(args.clean)
        clf.save(args.model_path)
        print(f"Saved {args.model_path}")
    if args.input is not None:
        clf = ProductClassifier.load(args.model_path)
        names = final_eval.load_product_names(args.fmt) if Path(args.fmt).exists() else {}
        df = pd.read_csv(args.input, low_memory=False)
        out = add_predictions(df, clf, names)
        output = args.output or args.input.with_name(args.input.stem + "_coded.csv")
        out.to_csv(output, index=False)
        msg = f"Wrote {len(out):,} rows to {output}"
        if "agree" in out.columns:
            msg += f" (GLiNER and classifier agree on {out['agree'].mean():.1%})"
        print(msg)


if __name__ == "__main__":
    main()
