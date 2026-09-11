"""Prepare a tiny AmazonHelp corpus without creating large local indexes.

The raw Kaggle CSV remains local. The script scans it in chunks and writes only
a bounded sample of customer -> AmazonHelp support pairs. This is enough for
intent discovery and the later golden-set workflow while avoiding large disk
usage on a developer machine.

Usage:
    python src/preprocess.py --input data/raw/twcs.csv

Outputs:
    data/processed/amazonhelp_pairs.jsonl
    data/processed/amazonhelp_stats.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

BRAND = "AmazonHelp"
COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "in_response_to_tweet_id",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract AmazonHelp support pairs")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--chunksize", type=int, default=100_000)
    p.add_argument("--max-pairs", type=int, default=5000)
    return p.parse_args()


def norm_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def clean_text(text: object) -> str:
    text = "" if pd.isna(text) else str(text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # First pass: map tweet IDs to compact customer tweet records. We only
    # retain inbound tweets because they can be parents of support replies.
    inbound: dict[str, dict] = {}
    total_rows = 0
    inbound_count = 0
    brand_count = 0

    print("=== Pass 1: collecting customer tweets ===")
    for chunk_no, chunk in enumerate(
        pd.read_csv(args.input, chunksize=args.chunksize, usecols=COLUMNS, low_memory=True),
        start=1,
    ):
        for row in chunk.itertuples(index=False):
            v = row._asdict()
            tweet_id = norm_id(v["tweet_id"])
            if not tweet_id:
                continue
            total_rows += 1
            is_inbound = str(v["inbound"]).lower() == "true"
            if is_inbound:
                inbound_count += 1
                inbound[tweet_id] = {
                    "tweet_id": tweet_id,
                    "author_id": str(v["author_id"]),
                    "created_at": str(v["created_at"]),
                    "text": clean_text(v["text"]),
                }
            if str(v["author_id"]) == BRAND:
                brand_count += 1
        if chunk_no % 10 == 0:
            print(f"Pass 1 scanned {total_rows:,} rows...", flush=True)

    print(f"Rows scanned: {total_rows:,}")
    print(f"Inbound customer tweets: {inbound_count:,}")
    print(f"AmazonHelp tweets: {brand_count:,}")
    print(f"Customer tweets retained for matching: {len(inbound):,}")

    # Second pass: stream AmazonHelp replies and keep only a bounded sample.
    # No response_tweet_id expansion is needed: the parent ID on the reply is
    # the cleanest direct customer -> support relation in TWCS.
    output = args.output_dir / "amazonhelp_pairs.jsonl"
    seen: set[str] = set()
    pair_count = 0
    intents = Counter()

    print("\n=== Pass 2: extracting customer -> AmazonHelp pairs ===")
    with output.open("w", encoding="utf-8") as out:
        for chunk_no, chunk in enumerate(
            pd.read_csv(args.input, chunksize=args.chunksize, usecols=COLUMNS, low_memory=True),
            start=1,
        ):
            for row in chunk.itertuples(index=False):
                v = row._asdict()
                if str(v["author_id"]) != BRAND:
                    continue
                parent_id = norm_id(v["in_response_to_tweet_id"])
                tweet_id = norm_id(v["tweet_id"])
                if not parent_id or not tweet_id or tweet_id in seen:
                    continue
                customer = inbound.get(parent_id)
                if customer is None:
                    continue

                record = {
                    "customer": customer,
                    "support": {
                        "tweet_id": tweet_id,
                        "author_id": BRAND,
                        "created_at": str(v["created_at"]),
                        "text": clean_text(v["text"]),
                    },
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen.add(tweet_id)
                pair_count += 1
                if pair_count >= args.max_pairs:
                    break
            if chunk_no % 10 == 0:
                print(f"Pass 2 scanned; extracted {pair_count:,} pairs...", flush=True)
            if pair_count >= args.max_pairs:
                break

    stats = {
        "brand": BRAND,
        "raw_rows_scanned": total_rows,
        "inbound_customer_tweets": inbound_count,
        "brand_tweets": brand_count,
        "customer_tweets_available_for_matching": len(inbound),
        "support_pairs_written": pair_count,
        "pair_sample_cap": args.max_pairs,
        "pair_definition": "AmazonHelp tweet whose in_response_to_tweet_id points to an inbound customer tweet",
    }
    stats_path = args.output_dir / "amazonhelp_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    print(json.dumps(stats, indent=2))
    print(f"\nWrote: {output}")
    print(f"Wrote: {stats_path}")


if __name__ == "__main__":
    main()
