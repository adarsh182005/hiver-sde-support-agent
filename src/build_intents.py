"""Discover candidate AmazonHelp intents with lightweight semantic clustering.

This exploratory step intentionally avoids heavyweight transformer/PyTorch
dependencies. Customer messages are cleaned, represented with TF-IDF, reduced
with LSA (TruncatedSVD), and clustered at several candidate K values. The
output is for human review before final intent labels are frozen.

Usage:
    python src/build_intents.py
    python src/build_intents.py --clusters 8 10 12 15
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discover candidate support intents")
    p.add_argument("--input", type=Path, default=Path("data/processed/amazonhelp_pairs.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--clusters", type=int, nargs="+", default=[8, 10, 12, 15])
    p.add_argument("--examples-per-cluster", type=int, default=8)
    p.add_argument("--components", type=int, default=100)
    return p.parse_args()


def clean(text: str) -> str:
    """Remove Twitter-specific noise while preserving support semantics."""
    text = str(text).replace("\u200b", " ").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@[a-z0-9_]+", " ", text)
    text = re.sub(r"\brt\b", " ", text)
    text = re.sub(r"[^\w$£€!?.,' -]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def is_usable(text: str) -> bool:
    """Exclude obvious acknowledgements/noise from taxonomy discovery only."""
    if len(text) < 12:
        return False
    words = re.findall(r"\b\w+\b", text)
    if len(words) < 3:
        return False
    normalized = " ".join(words)
    noise = {
        "thanks", "thank you", "thx", "ok", "okay", "great", "perfect",
        "awesome", "cool", "got it", "never mind", "nevermind",
    }
    return normalized not in noise


def representative_indices(
    embedding: np.ndarray, labels: np.ndarray, cluster_id: int, limit: int
) -> np.ndarray:
    indices = np.where(labels == cluster_id)[0]
    if not len(indices):
        return indices
    centroid = embedding[indices].mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-12)
    scores = embedding[indices] @ centroid
    return indices[np.argsort(scores)[::-1][:limit]]


def cluster_once(
    embeddings: np.ndarray,
    raw_texts: list[str],
    clean_texts: list[str],
    k: int,
    examples_per_cluster: int,
) -> dict:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(embeddings)

    sample_size = min(2000, len(embeddings))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(embeddings), size=sample_size, replace=False)
    silhouette = silhouette_score(
        embeddings[sample_idx], labels[sample_idx], metric="cosine"
    )

    clusters = []
    for cluster_id in range(k):
        indices = np.where(labels == cluster_id)[0]
        example_idx = representative_indices(
            embeddings, labels, cluster_id, examples_per_cluster
        )
        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": int(len(indices)),
                "examples": [raw_texts[i] for i in example_idx],
                "clean_examples": [clean_texts[i] for i in example_idx],
            }
        )

    clusters.sort(key=lambda x: x["size"], reverse=True)
    sizes = [c["size"] for c in clusters]
    return {
        "clusters": k,
        "silhouette_score_sample": round(float(silhouette), 4),
        "cluster_size_min": int(min(sizes)),
        "cluster_size_max": int(max(sizes)),
        "clusters_detail": clusters,
    }


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if any(k < 2 for k in args.clusters):
        raise ValueError("Every cluster count must be >= 2")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    raw_texts = [r["customer"]["text"] for r in records]
    cleaned_all = [clean(x) for x in raw_texts]

    keep = [is_usable(x) for x in cleaned_all]
    usable_raw = [x for x, ok in zip(raw_texts, keep) if ok]
    usable_clean = [x for x, ok in zip(cleaned_all, keep) if ok]
    if len(usable_clean) < max(args.clusters):
        raise ValueError(
            f"Need at least {max(args.clusters)} usable messages, found {len(usable_clean)}"
        )

    print("=== Intent discovery: TF-IDF + LSA ===")
    print(f"Input messages:    {len(records):,}")
    print(f"Usable messages:   {len(usable_clean):,}")
    print(f"Excluded noise:    {len(records) - len(usable_clean):,}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=20_000,
        stop_words="english",
    )
    X = vectorizer.fit_transform(usable_clean)
    print(f"TF-IDF features:   {X.shape[1]:,}")

    components = min(args.components, X.shape[1] - 1, len(usable_clean) - 1)
    svd = TruncatedSVD(n_components=components, random_state=42)
    embeddings = svd.fit_transform(X)
    embeddings = Normalizer(copy=False).fit_transform(embeddings).astype(np.float32)
    explained = float(svd.explained_variance_ratio_.sum())
    print(f"LSA components:    {components}")
    print(f"Variance retained: {explained:.4f}")

    results = []
    for k in sorted(set(args.clusters)):
        result = cluster_once(
            embeddings, usable_raw, usable_clean, k, args.examples_per_cluster
        )
        results.append(result)
        print(
            f"k={k:>2} | silhouette={result['silhouette_score_sample']:.4f} | "
            f"cluster sizes={result['cluster_size_min']}-{result['cluster_size_max']}"
        )

    results.sort(key=lambda x: x["silhouette_score_sample"], reverse=True)
    output = args.output_dir / "intent_candidates.json"
    payload = {
        "method": "TF-IDF (1-2 grams) + TruncatedSVD/LSA + KMeans",
        "random_state": 42,
        "input_messages": len(records),
        "usable_messages": len(usable_clean),
        "excluded_noise_messages": len(records) - len(usable_clean),
        "tfidf_features": int(X.shape[1]),
        "lsa_components": int(components),
        "lsa_variance_retained": round(explained, 4),
        "note": (
            "Exploratory candidate groups only. The highest silhouette score is "
            "not automatically the best business taxonomy; human review should "
            "favor coherent, actionable support intents and merge/split clusters "
            "when needed. Excluded acknowledgements remain in the source data."
        ),
        "candidate_runs": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote: {output}")
    print("\n=== Best candidate run ===")
    best = results[0]
    print(
        f"k={best['clusters']} | silhouette={best['silhouette_score_sample']:.4f} | "
        f"cluster sizes={best['cluster_size_min']}-{best['cluster_size_max']}"
    )
    for cluster in best["clusters_detail"]:
        print(f"\n[{cluster['cluster_id']}] size={cluster['size']}")
        for example in cluster["examples"][:5]:
            print(" -", example)


if __name__ == "__main__":
    main()
