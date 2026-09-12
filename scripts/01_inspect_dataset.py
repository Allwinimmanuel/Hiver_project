"""
scripts/01_inspect_dataset.py
------------------------------
Milestone 1 — Dataset Inspection

Reads the TWCS CSV in ~100k-row chunks so the full 5 GB file never
lands in RAM at once.  Collects all statistics incrementally and
writes two report files:

  reports/dataset_inspection.md
  reports/brand_summary.csv

Run with:
    py scripts/01_inspect_dataset.py
"""

import os
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DATASET_PATH = Path("twitter/twcs/twcs.csv")
REPORTS_DIR  = Path("reports")
CHUNK_SIZE   = 100_000      # rows per chunk — keeps RAM low
TOP_N        = 15           # brands to highlight in the report

EXPECTED_COLS = {
    "tweet_id", "author_id", "inbound",
    "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id",
}


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ──────────────────────────────────────────────
# STEP 1 — Verify file and print size
# ──────────────────────────────────────────────
section("STEP 1 — File verification")

if not DATASET_PATH.exists():
    print(f"[ERROR] Dataset not found: {DATASET_PATH.resolve()}")
    sys.exit(1)

file_bytes = DATASET_PATH.stat().st_size
print(f"[OK]  Path : {DATASET_PATH.resolve()}")
print(f"[OK]  Size : {human_size(file_bytes)}  ({file_bytes:,} bytes)")


# ──────────────────────────────────────────────
# STEP 2 — Peek at schema (first 10k rows only)
# ──────────────────────────────────────────────
section("STEP 2 — Schema inspection (first 10,000 rows)")

peek_iter = pd.read_csv(
    DATASET_PATH,
    chunksize=10_000,
    low_memory=False,
    on_bad_lines="warn",
)
peek_df = next(peek_iter)
del peek_iter          # release file handle immediately

columns = list(peek_df.columns)
dtypes  = peek_df.dtypes.to_dict()

print(f"Columns ({len(columns)}): {columns}")
print()
for col, dt in dtypes.items():
    print(f"  {col:<38} {dt}")

missing_expected = EXPECTED_COLS - set(columns)
if missing_expected:
    print(f"\n[WARN] Expected columns missing: {missing_expected}")
else:
    print("\n[OK]  All expected columns present.")


# ──────────────────────────────────────────────
# STEP 3 — Full chunked scan
# ──────────────────────────────────────────────
section("STEP 3 — Full dataset scan (chunked)")

total_rows       = 0
null_counts      = defaultdict(int)
inbound_count    = 0
outbound_count   = 0
inbound_null     = 0
tweet_id_counter = Counter()
brand_outbound   = Counter()   # author_id -> outbound msg count
author_counts    = Counter()   # author_id -> all msgs
tweets_are_reply = 0           # rows with in_response_to_tweet_id filled
tweets_got_reply = 0           # rows with response_tweet_id filled

chunk_num  = 0
start_time = time.time()

chunk_iter = pd.read_csv(
    DATASET_PATH,
    chunksize=CHUNK_SIZE,
    low_memory=False,
    on_bad_lines="warn",
    dtype=str,             # str avoids mixed-type inference issues
)

for chunk in chunk_iter:
    chunk_num  += 1
    total_rows += len(chunk)

    elapsed = time.time() - start_time
    print(
        f"  chunk {chunk_num:>4d}  |  rows: {total_rows:>10,}  |  "
        f"elapsed: {elapsed:>6.1f}s",
        end="\r",
        flush=True,
    )

    # Null counts
    for col in columns:
        if col in chunk.columns:
            null_counts[col] += int(chunk[col].isna().sum())

    # Duplicate tweet_id tracking
    if "tweet_id" in chunk.columns:
        tweet_id_counter.update(chunk["tweet_id"].dropna().tolist())

    # Inbound / outbound
    if "inbound" in chunk.columns:
        ib = chunk["inbound"].str.strip().str.lower()
        inbound_count  += int((ib == "true").sum())
        outbound_count += int((ib == "false").sum())
        inbound_null   += int(ib.isna().sum())

    # Author counts (all rows)
    if "author_id" in chunk.columns:
        author_counts.update(chunk["author_id"].dropna().tolist())

    # Brand outbound counts
    if "inbound" in chunk.columns and "author_id" in chunk.columns:
        ib = chunk["inbound"].str.strip().str.lower()
        out_authors = chunk.loc[ib == "false", "author_id"].dropna()
        brand_outbound.update(out_authors.tolist())

    # Conversation fields
    if "in_response_to_tweet_id" in chunk.columns:
        tweets_are_reply += int(chunk["in_response_to_tweet_id"].notna().sum())
    if "response_tweet_id" in chunk.columns:
        tweets_got_reply += int(chunk["response_tweet_id"].notna().sum())

print()   # newline after progress line
print(
    f"\n[OK]  Scan complete — {total_rows:,} rows, "
    f"{chunk_num} chunks, {time.time()-start_time:.1f}s"
)


# ──────────────────────────────────────────────
# STEP 4 — Derived statistics
# ──────────────────────────────────────────────
section("STEP 4 — Derived statistics")

null_pct        = {col: null_counts[col] / total_rows * 100 for col in columns}
dup_ids         = {tid: cnt for tid, cnt in tweet_id_counter.items() if cnt > 1}
total_unique    = len(tweet_id_counter)
total_dup_rows  = sum(cnt - 1 for cnt in dup_ids.values())
top_brands      = brand_outbound.most_common(TOP_N)

print(f"  Total rows           : {total_rows:,}")
print(f"  Unique tweet_ids     : {total_unique:,}")
print(f"  Duplicate tweet_ids  : {len(dup_ids):,}  ({total_dup_rows:,} extra rows)")
print(f"\n  Inbound  messages    : {inbound_count:,}")
print(f"  Outbound messages    : {outbound_count:,}")
print(f"  Inbound null rows    : {inbound_null:,}")

print(f"\n  Top {TOP_N} support accounts (by outbound message count):")
for rank, (brand, cnt) in enumerate(top_brands, 1):
    pct = cnt / total_rows * 100
    print(f"    {rank:>2}. {brand:<32} {cnt:>8,}  ({pct:.2f}%)")

quality_flags = []
if inbound_null > 0:
    quality_flags.append(
        f"{inbound_null:,} rows have NULL 'inbound' — direction unknown"
    )
if total_dup_rows > 0:
    quality_flags.append(
        f"{total_dup_rows:,} duplicate tweet_id rows (de-duplication recommended)"
    )
for col in columns:
    if null_pct.get(col, 0) > 50:
        quality_flags.append(
            f"Column '{col}' is {null_pct[col]:.1f}% null — sparse/optional field"
        )

if quality_flags:
    print("\n  [DATA QUALITY FLAGS]")
    for f in quality_flags:
        print(f"    WARNING: {f}")
else:
    print("\n  [OK] No major data quality issues.")


# ──────────────────────────────────────────────
# STEP 5 — Write brand_summary.csv
# ──────────────────────────────────────────────
section("STEP 5 — Writing reports/brand_summary.csv")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

brand_rows = [
    {
        "author_id":         brand,
        "outbound_messages": cnt,
        "total_author_msgs": author_counts.get(brand, 0),
    }
    for brand, cnt in brand_outbound.most_common()
]

brand_df = pd.DataFrame(brand_rows)
brand_csv = REPORTS_DIR / "brand_summary.csv"
brand_df.to_csv(brand_csv, index=False)
print(f"[OK]  {brand_csv}  ({len(brand_df):,} brands)")


# ──────────────────────────────────────────────
# STEP 6 — Write dataset_inspection.md
# ──────────────────────────────────────────────
section("STEP 6 — Writing reports/dataset_inspection.md")


def md_table(headers, rows):
    """Build a simple markdown table."""
    sep = "|".join("---" for _ in headers)
    lines = ["| " + " | ".join(headers) + " |", f"|{sep}|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


dtype_tbl = md_table(
    ["Column", "Dtype"],
    [[f"`{c}`", f"`{dtypes[c]}`"] for c in columns],
)

null_tbl = md_table(
    ["Column", "Null Count", "Null %"],
    [
        [f"`{c}`", f"{null_counts[c]:,}", f"{null_pct[c]:.2f}%"]
        for c in columns
    ],
)

brand_tbl = md_table(
    ["Rank", "Author ID", "Outbound Msgs", "% of Total"],
    [
        [rank, f"`{b}`", f"{cnt:,}", f"{cnt/total_rows*100:.2f}%"]
        for rank, (b, cnt) in enumerate(top_brands, 1)
    ],
)

qf_md = (
    "\n".join(f"- ⚠ {f}" for f in quality_flags)
    if quality_flags
    else "- ✅ No major issues detected."
)

best_brand, best_cnt = top_brands[0] if top_brands else ("N/A", 0)
best_pct = best_cnt / total_rows * 100 if total_rows else 0
runner_up = (
    f"Runner-up is `{top_brands[1][0]}` with {top_brands[1][1]:,} messages."
    if len(top_brands) > 1
    else ""
)

report = f"""# TWCS Dataset Inspection Report
Generated by: `scripts/01_inspect_dataset.py`

---

## 1. Dataset Size

| Property | Value |
|----------|-------|
| File path | `{DATASET_PATH}` |
| File size | {human_size(file_bytes)} ({file_bytes:,} bytes) |
| Total rows | {total_rows:,} |
| Chunk size used | {CHUNK_SIZE:,} rows |

---

## 2. Columns and Data Types

{dtype_tbl}

---

## 3. Missing Value Statistics

{null_tbl}

---

## 4. Duplicate Tweet ID Analysis

| Metric | Value |
|--------|-------|
| Unique `tweet_id` values | {total_unique:,} |
| `tweet_id` values appearing more than once | {len(dup_ids):,} |
| Extra (duplicate) rows | {total_dup_rows:,} |

{"✅ No duplicate tweet_ids detected." if not dup_ids else "⚠ Duplicates exist — de-duplicate before training."}

---

## 5. Inbound vs. Outbound Distribution

| Direction | Count | % of Total |
|-----------|-------|------------|
| Inbound (customer → brand) | {inbound_count:,} | {inbound_count/total_rows*100:.2f}% |
| Outbound (brand → customer) | {outbound_count:,} | {outbound_count/total_rows*100:.2f}% |
| Null / Unknown | {inbound_null:,} | {inbound_null/total_rows*100:.2f}% |

---

## 6. Top {TOP_N} Support Accounts

Ranked by **outbound** message count (how many replies the brand sent).
More outbound messages → more training data for that brand's AI agent.

{brand_tbl}

---

## 7. Conversation Statistics

| Metric | Value |
|--------|-------|
| Rows with `in_response_to_tweet_id` filled | {tweets_are_reply:,} |
| Rows with `response_tweet_id` filled | {tweets_got_reply:,} |

- `in_response_to_tweet_id` filled → this tweet is a reply to someone else.
- `response_tweet_id` filled → this tweet received at least one reply.
- These two fields together allow full conversation thread reconstruction.

---

## 8. Data Quality Observations

{qf_md}

---

## 9. Brand Recommendation

**Recommended brand: `{best_brand}`**

**Rationale:**
- Highest outbound volume: {best_cnt:,} messages ({best_pct:.2f}% of the entire dataset).
- More messages = more (customer-question, brand-answer) training pairs.
- A large corpus reduces risk of under-fitting or sparse coverage.
- {runner_up}

> **Next step (Milestone 2):** Filter to `author_id == "{best_brand}"` conversations,
> extract clean (question, answer) pairs, and prepare them for the AI agent.
"""

md_path = REPORTS_DIR / "dataset_inspection.md"
md_path.write_text(report, encoding="utf-8")
print(f"[OK]  {md_path}")


# ──────────────────────────────────────────────
# DONE
# ──────────────────────────────────────────────
section("DONE")
print(f"All reports saved to: {REPORTS_DIR.resolve()}")
print(f"  • dataset_inspection.md")
print(f"  • brand_summary.csv")
print(
    "\nReview the reports and confirm the recommended brand "
    "before proceeding to Milestone 2."
)
