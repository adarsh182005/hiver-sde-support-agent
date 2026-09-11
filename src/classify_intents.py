"""Train and evaluate a lightweight AmazonHelp intent classifier.

Uses TF-IDF word/character features + Logistic Regression so the pipeline stays
CPU-friendly and reproducible in a small take-home environment.

Important: the exploratory 5,000-pair sample does not contain final hand labels.
This script therefore supports a CSV/JSONL file with a manually reviewed
`intent` column. It refuses to silently invent labels.

Expected input columns:
  customer_text, intent

Example:
  python src/classify_intents.py --input data/processed/labeled_pairs.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


TAXONOMY = {
    "order_status_tracking",
    "delivery_delay",
    "delivery_attempt_problem",
    "payment_billing",
    "refund",
    "return_replacement",
    "order_change_cancellation",
    "prime_subscription",
    "product_item_issue",
    "account_technical_support",
    "other_ambiguous",
}


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError("Input must be .csv or .jsonl")

    required = {"customer_text", "intent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[["customer_text", "intent"]].dropna().copy()
    df["customer_text"] = df["customer_text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[df["customer_text"].ne("")]

    unknown = sorted(set(df["intent"]) - TAXONOMY)
    if unknown:
        raise ValueError(f"Unknown intents: {unknown}")
    return df


def build_model() -> Pipeline:
    # Word features capture phrases; character features help with misspellings,
    # usernames, product names, and short/noisy Twitter messages.
    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=30000,
                sublinear_tf=True,
            ),
        ),
    ])

    return Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
            ),
        ),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Manually labelled CSV/JSONL")
    parser.add_argument("--output", default="data/processed/intent_classifier_metrics.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_data(Path(args.input))
    counts = df["intent"].value_counts()
    if len(counts) < 2:
        raise ValueError("Need at least two intent classes")
    if counts.min() < 2:
        raise ValueError("Every class needs at least 2 examples for a stratified split")

    X_train, X_test, y_train, y_test = train_test_split(
        df["customer_text"],
        df["intent"],
        test_size=args.test_size,
        random_state=42,
        stratify=df["intent"],
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    confidence = probabilities.max(axis=1)

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
        "test_class_distribution": counts.to_dict(),
        "classification_report": report,
        "confidence_summary": {
            "mean": float(np.mean(confidence)),
            "median": float(np.median(confidence)),
            "p10": float(np.quantile(confidence, 0.10)),
            "p90": float(np.quantile(confidence, 0.90)),
        },
        "split": "80/20 stratified, random_state=42",
        "model": "TF-IDF word+character n-grams + balanced Logistic Regression",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=== AmazonHelp intent classifier ===")
    print(f"Rows:       {len(df):,}")
    print(f"Train/test: {len(X_train):,} / {len(X_test):,}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Macro F1:   {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:{metrics['weighted_f1']:.4f}")
    print(f"Mean conf.: {metrics['confidence_summary']['mean']:.4f}")
    print(f"Wrote:      {output}")


if __name__ == "__main__":
    main()
