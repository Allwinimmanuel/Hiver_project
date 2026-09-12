"""
scripts/02_extract_amazonhelp.py
---------------------------------
Milestone 2, Task 1 — Extract AmazonHelp Data

Strategy (2-pass, memory-safe):
  Pass 1 : Stream the full CSV. Collect every AmazonHelp OUTBOUND row.
           Record the tweet_ids those rows were replying to
           (in_response_to_tweet_id).
  Pass 2 : Stream again. Collect every row whose tweet_id appears in
           the set collected in Pass 1 (= customer messages).
  Merge  : Combine both sets and save to data/processed/amazonhelp_raw.csv

Why 2 passes?
  We cannot know which customer tweets belong to AmazonHelp until we
  have seen which tweet_ids AmazonHelp replied to.  Storing only a set
  of integer tweet_ids uses negligible RAM.

Output : data/processed/amazonhelp_raw.csv
"""

import sys
import time
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────
DATASET_PATH = Path("twitter/twcs/twcs.csv")
OUT_PATH     = Path("data/processed/amazonhelp_raw.csv")
CHUNK_SIZE   = 100_000
BRAND        = "AmazonHelp"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── Guard ────────────────────────────────────────────────────
if not DATASET_PATH.exists():
    print(f"[ERROR] Dataset not found: {DATASET_PATH.resolve()}")
    sys.exit(1)

# ════════════════════════════════════════════════════════════
# PASS 1 — Collect AmazonHelp outbound rows
#           and the set of parent tweet_ids they replied to
# ════════════════════════════════════════════════════════════
section(f"PASS 1 — Collecting {BRAND} outbound rows")

amazon_rows   = []   # list of dicts – only ~170 k rows, fine in RAM
parent_ids    = set()   # tweet_ids AmazonHelp replied to (customer tweets)
total_scanned = 0
chunk_num     = 0
t0            = time.time()

for chunk in pd.read_csv(
    DATASET_PATH,
    chunksize=CHUNK_SIZE,
    low_memory=False,
    on_bad_lines="warn",
):
    chunk_num     += 1
    total_scanned += len(chunk)

    # Filter: rows where AmazonHelp is the author AND inbound is False
    # inbound column is bool in this dataset
    mask = (chunk["author_id"] == BRAND) & (chunk["inbound"] == False)  # noqa: E712
    ah_chunk = chunk[mask]

    if len(ah_chunk) > 0:
        amazon_rows.append(ah_chunk)
        # Collect parent tweet_ids (as integers where possible)
        parents = (
            ah_chunk["in_response_to_tweet_id"]
            .dropna()
            .astype(int)
            .tolist()
        )
        parent_ids.update(parents)

    print(
        f"  chunk {chunk_num:>3d} | scanned {total_scanned:>10,} | "
        f"AH rows: {sum(len(r) for r in amazon_rows):>7,} | "
        f"parent IDs: {len(parent_ids):>7,}",
        end="\r", flush=True,
    )

print()  # newline after \r

amazon_df = pd.concat(amazon_rows, ignore_index=True) if amazon_rows else pd.DataFrame()
print(f"\n[OK]  AmazonHelp outbound rows : {len(amazon_df):,}")
print(f"[OK]  Unique parent tweet_ids  : {len(parent_ids):,}")
print(f"[OK]  Pass 1 elapsed           : {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# PASS 2 — Collect customer tweets that AmazonHelp replied to
# ════════════════════════════════════════════════════════════
section("PASS 2 — Collecting customer (parent) tweets")

customer_rows = []
chunk_num     = 0
t1            = time.time()

for chunk in pd.read_csv(
    DATASET_PATH,
    chunksize=CHUNK_SIZE,
    low_memory=False,
    on_bad_lines="warn",
):
    chunk_num += 1

    # Match rows whose tweet_id is in our parent_ids set
    # tweet_id is int64; parent_ids is a set of Python ints
    mask = chunk["tweet_id"].isin(parent_ids)
    cust_chunk = chunk[mask]

    if len(cust_chunk) > 0:
        customer_rows.append(cust_chunk)

    print(
        f"  chunk {chunk_num:>3d} | customer rows found: "
        f"{sum(len(r) for r in customer_rows):>7,}",
        end="\r", flush=True,
    )

print()

customer_df = pd.concat(customer_rows, ignore_index=True) if customer_rows else pd.DataFrame()
print(f"\n[OK]  Customer rows found : {len(customer_df):,}")
print(f"[OK]  Pass 2 elapsed      : {time.time()-t1:.1f}s")


# ════════════════════════════════════════════════════════════
# MERGE & SAVE
# ════════════════════════════════════════════════════════════
section("MERGE — Combining and saving")

combined = pd.concat([amazon_df, customer_df], ignore_index=True)

# Drop rows with duplicate tweet_id (shouldn't happen, but be safe)
before = len(combined)
combined.drop_duplicates(subset=["tweet_id"], inplace=True)
after  = len(combined)
print(f"  Combined rows          : {before:,}")
print(f"  After dedup            : {after:,}  (removed {before-after:,} duplicates)")

# Sort by tweet_id for reproducibility
combined.sort_values("tweet_id", inplace=True)
combined.reset_index(drop=True, inplace=True)

combined.to_csv(OUT_PATH, index=False)
size_mb = OUT_PATH.stat().st_size / 1024 / 1024
print(f"\n[OK]  Saved: {OUT_PATH}  ({len(combined):,} rows, {size_mb:.2f} MB)")
print(f"[OK]  Columns: {list(combined.columns)}")
print(f"\n  AmazonHelp outbound rows : {(combined['author_id'] == BRAND).sum():,}")
print(f"  Customer inbound rows    : {(combined['author_id'] != BRAND).sum():,}")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
section("DONE — Task 1 complete")
