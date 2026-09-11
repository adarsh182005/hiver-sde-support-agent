"""Quick reconnaissance of Customer Support on Twitter.

Reads the large CSV in chunks so the full dataset does not need to fit in memory.
The first pass ranks likely support brands by the number of outbound tweets
(`inbound == False`).

Usage:
    python src/sample_brand.py --input data/raw/twcs.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
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


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Dataset not found: {args.input}")

    outbound = Counter()
    inbound = Counter()
    total_rows = 0
    total_inbound = 0
    total_outbound = 0

    reader = pd.read_csv(
        args.input,
        chunksize=args.chunksize,
        usecols=lambda column: column in REQUIRED_COLUMNS,
        low_memory=True,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        inbound_mask = chunk["inbound"].astype(str).str.lower().eq("true")

        inbound.update(chunk.loc[inbound_mask, "author_id"].astype(str))
        outbound.update(chunk.loc[~inbound_mask, "author_id"].astype(str))
        total_inbound += int(inbound_mask.sum())
        total_outbound += int((~inbound_mask).sum())

        if chunk_number % 10 == 0:
            print(f"Processed {total_rows:,} rows...", flush=True)

    print("\n=== Dataset reconnaissance ===")
    print(f"Rows scanned:       {total_rows:,}")
    print(f"Inbound tweets:     {total_inbound:,}")
    print(f"Outbound tweets:    {total_outbound:,}")
    print(f"Unique authors:     {len(set(inbound) | set(outbound)):,}")

    print(f"\n=== Top {args.top} likely support brands ===")
    print(f"{'Rank':<5} {'Brand':<25} {'Replies':>10} {'Customer tweets':>16}")
    print("-" * 62)

    for rank, (brand, reply_count) in enumerate(outbound.most_common(args.top), start=1):
        customer_count = inbound.get(brand, 0)
        print(f"{rank:<5} {brand:<25} {reply_count:>10,} {customer_count:>16,}")


if __name__ == "__main__":
    main()
