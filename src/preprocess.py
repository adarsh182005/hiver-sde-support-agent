"""Prepare a compact AmazonHelp support corpus without a full-dataset index.

The raw Kaggle CSV stays local and is scanned in chunks. We first collect the
IDs of customer tweets directly answered by AmazonHelp, then make a second
pass and keep only those customer tweets plus AmazonHelp replies. This avoids
creating a multi-million-row SQLite database and keeps disk usage small.

Usage:
    python src/preprocess.py --input data/raw/twcs.csv

Outputs:
    data/processed/amazonhelp_messages.jsonl
    data/processed/amazonhelp_stats.json
"""

from __future__ import annotations

import argparse
import json
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
    "response_tweet_id",
    "in_response_to_tweet_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AmazonHelp support data")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-messages", type=int, default=350_000)
    return parser.parse_args()


def norm_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def parse_ids(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [x for x in (norm_id(v) for v in str(value).split(",")) if x]


def read_chunks(path: Path, chunksize: int):
    return pd.read_csv(path, chunksize=chunksize, usecols=COLUMNS, low_memory=True)


def collect_targets(csv_path: Path, chunksize: int) -> tuple[set[str], int, int]:
    """Collect customer parent IDs directly answered by AmazonHelp."""
    answered_customer_ids: set[str] = set()
    brand_tweets = 0
    total_rows = 0

    for chunk_no, chunk in enumerate(read_chunks(csv_path, chunksize), start=1):
        for row in chunk.itertuples(index=False):
            values = row._asdict()
            tweet_id = norm_id(values["tweet_id"])
            author = str(values["author_id"])
            if not tweet_id:
                continue
            total_rows += 1
            if author == BRAND:
                brand_tweets += 1
                parent = norm_id(values["in_response_to_tweet_id"])
                if parent:
                    answered_customer_ids.add(parent)
        if chunk_no % 10 == 0:
            print(f"Pass 1 scanned {total_rows:,} rows...", flush=True)

    return answered_customer_ids, total_rows, brand_tweets


def write_compact_corpus(
    csv_path: Path,
    output_path: Path,
    target_ids: set[str],
    chunksize: int,
    max_messages: int,
) -> tuple[int, Counter]:
    """Second pass: retain AmazonHelp replies and their directly answered parents."""
    seen: set[str] = set()
    counts: Counter = Counter()
    written = 0

    with output_path.open("w", encoding="utf-8") as out:
        for chunk_no, chunk in enumerate(read_chunks(csv_path, chunksize), start=1):
            for row in chunk.itertuples(index=False):
                values = row._asdict()
                tweet_id = norm_id(values["tweet_id"])
                author = str(values["author_id"])
                if not tweet_id or tweet_id in seen:
                    continue

                keep = author == BRAND or tweet_id in target_ids
                if not keep:
                    continue

                record = {
                    "tweet_id": tweet_id,
                    "author_id": author,
                    "inbound": str(values["inbound"]).lower() == "true",
                    "created_at": str(values["created_at"]),
                    "text": str(values["text"]),
                    "in_response_to_tweet_id": norm_id(values["in_response_to_tweet_id"]),
                    "response_tweet_ids": parse_ids(values["response_tweet_id"]),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen.add(tweet_id)
                written += 1
                counts["brand_messages"] += author == BRAND
                counts["customer_messages"] += author != BRAND

                if written >= max_messages:
                    break
            if chunk_no % 10 == 0:
                print(f"Pass 2 scanned; wrote {written:,} relevant messages...", flush=True)
            if written >= max_messages:
                break

    return written, counts


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output_dir / "amazonhelp_messages.jsonl"
    stats_path = args.output_dir / "amazonhelp_stats.json"

    print("=== Pass 1: finding AmazonHelp customer messages ===")
    target_ids, total_rows, brand_tweets = collect_targets(args.input, args.chunksize)
    print(f"Rows scanned: {total_rows:,}")
    print(f"AmazonHelp tweets: {brand_tweets:,}")
    print(f"Unique customer tweets answered: {len(target_ids):,}")

    print("\n=== Pass 2: writing compact support corpus ===")
    written, counts = write_compact_corpus(
        args.input, output_path, target_ids, args.chunksize, args.max_messages
    )

    # Conversation-level statistics are intentionally based on direct
    # customer -> AmazonHelp links, which are reliable in the TWCS schema.
    stats = {
        "brand": BRAND,
        "raw_rows_scanned": total_rows,
        "brand_tweets": brand_tweets,
        "unique_customer_tweets_answered": len(target_ids),
        "compact_messages_written": written,
        "customer_messages_written": counts["customer_messages"],
        "brand_messages_written": counts["brand_messages"],
        "corpus_definition": "AmazonHelp tweets plus customer tweets directly referenced by AmazonHelp's in_response_to_tweet_id",
        "max_messages": args.max_messages,
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    print(json.dumps(stats, indent=2))
    print(f"\nWrote: {output_path}")
    print(f"Wrote: {stats_path}")


if __name__ == "__main__":
    main()
