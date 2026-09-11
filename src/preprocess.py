"""Extract and reconstruct AmazonHelp support conversations.

The raw Kaggle dataset is kept local. This script builds a small SQLite index
from the CSV, finds AmazonHelp tweets, reconstructs connected conversation
threads through parent/child tweet links, and writes a compact JSONL sample.

Usage:
    python src/preprocess.py --input data/raw/twcs.csv

Outputs:
    data/processed/amazonhelp_conversations.jsonl
    data/processed/amazonhelp_stats.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, deque
from pathlib import Path

import pandas as pd

BRAND = "AmazonHelp"
REQUIRED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AmazonHelp conversations")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-conversations", type=int, default=5000)
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
    text = str(value).strip()
    if not text:
        return []
    return [x for x in (norm_id(v) for v in text.split(",")) if x]


def build_index(csv_path: Path, db_path: Path, chunksize: int) -> tuple[int, int]:
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute(
        """CREATE TABLE tweets (
            tweet_id TEXT PRIMARY KEY,
            author_id TEXT NOT NULL,
            inbound INTEGER NOT NULL,
            created_at TEXT,
            text TEXT,
            parent_id TEXT,
            response_ids TEXT
        )"""
    )
    conn.execute("CREATE INDEX idx_parent ON tweets(parent_id)")
    conn.execute("CREATE INDEX idx_author ON tweets(author_id)")

    total = 0
    brand_tweets = 0
    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        usecols=lambda c: c in REQUIRED_COLUMNS,
        low_memory=True,
    )

    for chunk_no, chunk in enumerate(reader, start=1):
        rows = []
        for row in chunk.itertuples(index=False):
            values = row._asdict()
            tweet_id = norm_id(values["tweet_id"])
            author = str(values["author_id"])
            if not tweet_id:
                continue
            inbound = str(values["inbound"]).lower() == "true"
            if author == BRAND:
                brand_tweets += 1
            rows.append(
                (
                    tweet_id,
                    author,
                    int(inbound),
                    str(values["created_at"]),
                    str(values["text"]),
                    norm_id(values["in_response_to_tweet_id"]),
                    json.dumps(parse_ids(values["response_tweet_id"])),
                )
            )
        conn.executemany(
            "INSERT OR REPLACE INTO tweets VALUES (?, ?, ?, ?, ?, ?, ?)", rows
        )
        conn.commit()
        total += len(rows)
        if chunk_no % 10 == 0:
            print(f"Indexed {total:,} rows...", flush=True)

    conn.close()
    return total, brand_tweets


def reconstruct(db_path: Path, output_path: Path, max_conversations: int) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    brand_ids = [
        row[0]
        for row in conn.execute(
            "SELECT tweet_id FROM tweets WHERE author_id = ?", (BRAND,)
        )
    ]

    # Conversation membership is the connected component of each AmazonHelp
    # tweet when following parent links and response_tweet_id child links.
    visited: set[str] = set()
    conversations: list[list[dict]] = []

    for seed in brand_ids:
        if seed in visited:
            continue
        queue = deque([seed])
        component: set[str] = set()

        while queue:
            tweet_id = queue.popleft()
            if tweet_id in component:
                continue
            row = conn.execute(
                "SELECT * FROM tweets WHERE tweet_id = ?", (tweet_id,)
            ).fetchone()
            if row is None:
                continue
            component.add(tweet_id)

            parent = row["parent_id"]
            if parent and parent not in component:
                queue.append(parent)

            for child in json.loads(row["response_ids"] or "[]"):
                if child not in component:
                    queue.append(child)

            # Parent links alone are enough for the common case, but querying
            # children makes reconstruction robust when response IDs are noisy.
            for child_row in conn.execute(
                "SELECT tweet_id FROM tweets WHERE parent_id = ?", (tweet_id,)
            ):
                child = child_row[0]
                if child not in component:
                    queue.append(child)

        visited.update(component)
        rows = [
            dict(conn.execute("SELECT * FROM tweets WHERE tweet_id = ?", (tid,)).fetchone())
            for tid in component
        ]
        rows.sort(key=lambda r: r["created_at"] or "")
        brand_count = sum(r["author_id"] == BRAND for r in rows)
        inbound_count = sum(bool(r["inbound"]) for r in rows)
        if brand_count and inbound_count:
            conversations.append(rows)
        if len(conversations) >= max_conversations:
            break

    with output_path.open("w", encoding="utf-8") as f:
        for rows in conversations:
            clean_rows = [
                {
                    "tweet_id": r["tweet_id"],
                    "author_id": r["author_id"],
                    "inbound": bool(r["inbound"]),
                    "created_at": r["created_at"],
                    "text": r["text"],
                    "in_response_to_tweet_id": r["parent_id"],
                }
                for r in rows
            ]
            f.write(json.dumps({"messages": clean_rows}, ensure_ascii=False) + "\n")

    lengths = [len(c) for c in conversations]
    stats = {
        "brand": BRAND,
        "conversation_count": len(conversations),
        "messages_in_conversations": sum(lengths),
        "median_messages_per_conversation": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "max_messages_per_conversation": max(lengths, default=0),
        "multi_turn_conversations": sum(x >= 3 for x in lengths),
        "inbound_messages": sum(sum(bool(r["inbound"]) for r in c) for c in conversations),
        "outbound_messages": sum(sum(not bool(r["inbound"]) for r in c) for c in conversations),
    }
    conn.close()
    return stats


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.output_dir / "twcs_index.sqlite"
    output_path = args.output_dir / "amazonhelp_conversations.jsonl"
    stats_path = args.output_dir / "amazonhelp_stats.json"

    print("=== Building local tweet index ===")
    total, brand_tweets = build_index(args.input, db_path, args.chunksize)
    print(f"Indexed rows: {total:,}")
    print(f"AmazonHelp tweets: {brand_tweets:,}")

    print("\n=== Reconstructing AmazonHelp conversations ===")
    stats = reconstruct(db_path, output_path, args.max_conversations)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"\nWrote: {output_path}")
    print(f"Wrote: {stats_path}")


if __name__ == "__main__":
    main()
