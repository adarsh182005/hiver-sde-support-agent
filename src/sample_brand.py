"""Quick reconnaissance of Customer Support on Twitter.

Reads the large CSV in chunks so the full dataset does not need to fit in memory.
Ranks candidate support brands using outbound support volume and the number of
unique inbound customer tweets directly answered by each support account.

Usage:
    python src/sample_brand.py --input data/raw/twcs.csv
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank support brands in twcs.csv")
    parser.add_argument("--input", type=Path, required=True, help="Path to twcs.csv")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def normalized_ids(series: pd.Series) -> pd.Series:
    """Normalize CSV numeric IDs so 3 and 3.0 compare as the same tweet ID."""
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Dataset not found: {args.input}")

    outbound = Counter()
    total_rows = 0
    total_inbound = 0
    total_outbound = 0

    # Pass 1: collect all inbound/customer tweet IDs. This lets us verify that
    # an outbound tweet really answers a customer message.
    inbound_ids: set[str] = set()
    reader = pd.read_csv(
        args.input,
        chunksize=args.chunksize,
        usecols=lambda column: column in REQUIRED_COLUMNS,
        low_memory=True,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        inbound_mask = chunk["inbound"].astype(str).str.lower().eq("true")
        ids = normalized_ids(chunk.loc[inbound_mask, "tweet_id"]).dropna()
        inbound_ids.update(ids.tolist())
        total_inbound += int(inbound_mask.sum())
        total_rows += len(chunk)

        if chunk_number % 10 == 0:
            print(f"Pass 1 processed {total_rows:,} rows...", flush=True)

    # Pass 2: rank support accounts and count unique customer tweets they answer.
    answered_by_brand: dict[str, set[str]] = defaultdict(set)
    total_rows_pass2 = 0
    reader = pd.read_csv(
        args.input,
        chunksize=args.chunksize,
        usecols=lambda column: column in REQUIRED_COLUMNS,
        low_memory=True,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        inbound_mask = chunk["inbound"].astype(str).str.lower().eq("true")
        outbound_mask = ~inbound_mask
        outbound_rows = chunk.loc[
            outbound_mask, ["author_id", "in_response_to_tweet_id"]
        ].dropna(subset=["author_id", "in_response_to_tweet_id"])

        outbound.update(chunk.loc[outbound_mask, "author_id"].astype(str))
        total_outbound += int(outbound_mask.sum())

        parent_ids = normalized_ids(outbound_rows["in_response_to_tweet_id"])
        for brand, parent_id in zip(
            outbound_rows["author_id"].astype(str), parent_ids
        ):
            if pd.notna(parent_id) and parent_id in inbound_ids:
                answered_by_brand[brand].add(str(parent_id))

        total_rows_pass2 += len(chunk)
        if chunk_number % 10 == 0:
            print(f"Pass 2 processed {total_rows_pass2:,} rows...", flush=True)

    print("\n=== Dataset reconnaissance ===")
    print(f"Rows scanned:       {total_rows:,}")
    print(f"Inbound tweets:     {total_inbound:,}")
    print(f"Outbound tweets:    {total_outbound:,}")
    print(f"Unique outbound authors: {len(outbound):,}")

    print(f"\n=== Top {args.top} likely support brands ===")
    print(f"{'Rank':<5} {'Brand':<25} {'Replies':>10} {'Customer tweets answered':>25}")
    print("-" * 70)

    for rank, (brand, reply_count) in enumerate(outbound.most_common(args.top), start=1):
        customer_count = len(answered_by_brand.get(brand, set()))
        print(f"{rank:<5} {brand:<25} {reply_count:>10,} {customer_count:>25,}")


if __name__ == "__main__":
    main()
