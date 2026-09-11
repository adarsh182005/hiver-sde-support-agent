"""Quick reconnaissance of Customer Support on Twitter.

Reads the large CSV in chunks so the full dataset does not need to fit in memory.
Ranks candidate support brands using outbound support volume and the number of
customer messages those support accounts directly replied to.

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
    replied_customer_tweets = Counter()
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
        outbound_mask = ~inbound_mask

        outbound_authors = chunk.loc[outbound_mask, "author_id"].astype(str)
        outbound.update(outbound_authors)

        # For an outbound support tweet, in_response_to_tweet_id points to the
        # customer tweet it is answering. Count these links per support account.
        replied = chunk.loc[outbound_mask, ["author_id", "in_response_to_tweet_id"]].copy()
        replied = replied.dropna(subset=["in_response_to_tweet_id"])
        replied_customer_tweets.update(replied["author_id"].astype(str))

        total_inbound += int(inbound_mask.sum())
        total_outbound += int(outbound_mask.sum())

        if chunk_number % 10 == 0:
            print(f"Processed {total_rows:,} rows...", flush=True)

    print("\n=== Dataset reconnaissance ===")
    print(f"Rows scanned:       {total_rows:,}")
    print(f"Inbound tweets:     {total_inbound:,}")
    print(f"Outbound tweets:    {total_outbound:,}")
    print(f"Unique authors:     {len(outbound):,} outbound authors")

    print(f"\n=== Top {args.top} likely support brands ===")
    print(f"{'Rank':<5} {'Brand':<25} {'Replies':>10} {'Customer tweets answered':>25}")
    print("-" * 70)

    for rank, (brand, reply_count) in enumerate(outbound.most_common(args.top), start=1):
        customer_count = replied_customer_tweets.get(brand, 0)
        print(f"{rank:<5} {brand:<25} {reply_count:>10,} {customer_count:>25,}")


if __name__ == "__main__":
    main()
