"""Discover candidate AmazonHelp intents from the extracted support pairs.

This is an exploratory, reproducible step—not the final taxonomy. It uses
TF-IDF features and MiniBatchKMeans to surface recurring customer-message
patterns. Human review will turn the clusters into the final intent labels.

Usage:
    python src/build_intents.py
    python src/build_intents.py --input data/processed/amazonhelp_pairs.jsonl --clusters 15
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discover candidate support intents")
    p.add_argument("--input", type=Path, default=Path("data/processed/amazonhelp_pairs.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--clusters", type=int, default=15)
    p.add_argument("--examples-per-cluster", type=int, default=12)
    return p.parse_args()


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"[^a-z0-9$£€ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    customer_texts = [r["customer"]["text"] for r in records]
    texts = [clean(x) for x in customer_texts]

    if len(texts) < args.clusters:
        raise ValueError(f"Need at least {args.clusters} messages, found {len(texts)}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=20_000,
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)

    model = MiniBatchKMeans(
        n_clusters=args.clusters,
        random_state=42,
        n_init=10,
        batch_size=256,
    )
    labels = model.fit_predict(X)

    # Silhouette is calculated on a bounded sample to keep runtime/memory
    # predictable on a laptop.
    sample_size = min(2000, X.shape[0])
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(X.shape[0], size=sample_size, replace=False)
    silhouette = silhouette_score(X[sample_idx], labels[sample_idx])

    terms = np.asarray(vectorizer.get_feature_names_out())
    centers = model.cluster_centers_
    clusters = []

    for cluster_id in range(args.clusters):
        indices = np.where(labels == cluster_id)[0]
        # Distance to centroid gives representative messages.
        if len(indices):
            distances = model.transform(X[indices])[:, cluster_id]
            order = np.argsort(distances)[: args.examples_per_cluster]
            example_indices = indices[order]
        else:
            example_indices = []

        top_terms = terms[np.argsort(centers[cluster_id])[-12:][::-1]].tolist()
        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": int(len(indices)),
                "top_terms": top_terms,
                "examples": [customer_texts[i] for i in example_indices],
            }
        )

    clusters.sort(key=lambda x: x["size"], reverse=True)
    result = {
        "method": "TF-IDF (1-2 grams) + MiniBatchKMeans",
        "random_state": 42,
        "messages": len(records),
        "clusters": args.clusters,
        "tfidf_features": int(X.shape[1]),
        "silhouette_score_sample": round(float(silhouette), 4),
        "note": "Clusters are candidate intent groups and require human review before becoming the final taxonomy.",
        "clusters_detail": clusters,
    }

    output = args.output_dir / "intent_candidates.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Intent discovery ===")
    print(f"Messages:          {len(records):,}")
    print(f"TF-IDF features:   {X.shape[1]:,}")
    print(f"Clusters:          {args.clusters}")
    print(f"Silhouette sample: {silhouette:.4f}")
    print(f"\nWrote: {output}")
    print("\n=== Candidate clusters ===")
    for cluster in clusters:
        print(f"\n[{cluster['cluster_id']}] size={cluster['size']}")
        print("Terms:", ", ".join(cluster["top_terms"][:8]))
        for example in cluster["examples"][:5]:
            print(" -", example)


if __name__ == "__main__":
    main()
