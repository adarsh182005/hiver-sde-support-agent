"""Create a human-review annotation queue from AmazonHelp support pairs.

The script does not assign intents automatically. It creates a stratified,
reproducible review sample using the exploratory cluster IDs when available,
while preserving the original customer message and historical support reply.

Usage:
    python src/create_annotation_queue.py
    python src/create_annotation_queue.py --n 500

Output:
    data/processed/annotation_queue.csv

After review, fill the `intent` column using the labels in
`data/processed/intent_taxonomy.json`. Keep the `reviewer_notes` column for
ambiguous cases and escalation rationale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_PATH = Path("data/processed/amazonhelp_pairs.jsonl")
CLUSTER_PATH = Path("data/processed/intent_candidates.json")
DEFAULT_OUTPUT = Path("data/processed/annotation_queue.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create AmazonHelp annotation queue")
    p.add_argument("--input", type=Path, default=PAIR_PATH)
    p.add_argument("--clusters", type=Path, default=CLUSTER_PATH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--n", type=int, default=500, help="Number of examples to review")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(
                {
                    "customer_tweet_id": record["customer"]["tweet_id"],
                    "customer_created_at": record["customer"]["created_at"],
                    "customer_text": record["customer"]["text"],
                    "support_tweet_id": record["support"]["tweet_id"],
                    "support_created_at": record["support"]["created_at"],
                    "support_reply": record["support"]["text"],
                }
            )
    return pd.DataFrame(rows)


def load_cluster_assignments(path: Path) -> dict[str, int]:
    """Return message -> exploratory cluster for the selected K run.

    The clustering artifact has changed over iterations, so this function is
    intentionally defensive. If assignments cannot be recovered, sampling
    falls back to deterministic random sampling rather than inventing labels.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    # Supported shapes: a flat list of assignments, or a run containing rows
    # with customer_text/message and cluster fields.
    assignments: dict[str, int] = {}
    candidates = data.get("assignments") if isinstance(data, dict) else data
    if isinstance(candidates, list):
        for item in candidates:
            if not isinstance(item, dict):
                continue
            text = item.get("customer_text") or item.get("message") or item.get("text")
            cluster = item.get("cluster")
            if text and isinstance(cluster, (int, float)):
                assignments[str(text).strip()] = int(cluster)
    return assignments


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    df = load_pairs(args.input)
    if df.empty:
        raise ValueError("No support pairs found")

    # Deduplicate by customer message where possible. We want the annotation
    # set to represent distinct customer requests, not repeated wording.
    df = df.drop_duplicates(subset=["customer_text"], keep="first").reset_index(drop=True)
    n = min(args.n, len(df))

    cluster_map = load_cluster_assignments(args.clusters)
    df["exploratory_cluster"] = df["customer_text"].map(cluster_map)

    if df["exploratory_cluster"].notna().any():
        # Allocate approximately equal review slots across available clusters,
        # with any remainder filled by a deterministic random sample. These
        # clusters are sampling strata only; they are NOT ground-truth labels.
        strata = []
        usable = df.dropna(subset=["exploratory_cluster"])
        clusters = sorted(usable["exploratory_cluster"].unique())
        base = n // len(clusters)
        remainder = n - base * len(clusters)
        for i, cluster in enumerate(clusters):
            group = usable[usable["exploratory_cluster"] == cluster]
            take = min(len(group), base + (1 if i < remainder else 0))
            if take:
                idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
                strata.extend(idx.tolist())
        selected = df.loc[strata].copy()
        if len(selected) < n:
            remaining = df.drop(index=selected.index)
            extra = rng.choice(remaining.index.to_numpy(), size=min(n - len(selected), len(remaining)), replace=False)
            selected = pd.concat([selected, df.loc[extra]], ignore_index=True)
    else:
        selected_idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
        selected = df.loc[selected_idx].copy()

    selected = selected.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    selected.insert(0, "annotation_id", [f"AQ-{i:04d}" for i in range(1, len(selected) + 1)])
    selected["intent"] = ""
    selected["reviewer_notes"] = ""
    selected["escalation_expected"] = ""
    selected["golden_eval"] = ""

    columns = [
        "annotation_id",
        "customer_tweet_id",
        "customer_created_at",
        "customer_text",
        "support_tweet_id",
        "support_created_at",
        "support_reply",
        "exploratory_cluster",
        "intent",
        "reviewer_notes",
        "escalation_expected",
        "golden_eval",
    ]
    selected = selected[columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("=== Annotation queue created ===")
    print(f"Unique customer messages available: {len(df):,}")
    print(f"Examples selected for review:       {len(selected):,}")
    print(f"Seed:                                {args.seed}")
    print(f"Cluster stratification available:   {df['exploratory_cluster'].notna().any()}")
    print(f"Wrote:                               {args.output}")
    print("\nNext: open the CSV and fill `intent` using the frozen taxonomy.")
    print("Use `other_ambiguous` when the message does not fit cleanly; record why in reviewer_notes.")


if __name__ == "__main__":
    main()
