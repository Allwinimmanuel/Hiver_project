"""
scripts/03_reconstruct_conversations.py
----------------------------------------
Milestone 2, Task 2 — Conversation Reconstruction

Loads data/processed/amazonhelp_raw.csv (small, ~340k rows, fits in RAM).

How conversation reconstruction works:
  - Each AmazonHelp outbound tweet has in_response_to_tweet_id pointing
    to the tweet it replied to (the customer message).
  - We pair: customer_tweet <-> amazon_reply using that field.
  - To assign a conversation_id, we walk UP the chain to find the root
    (the first customer tweet that started the thread).
  - conversation_turns = number of (customer, amazon) pairs in the thread.

Output columns:
  conversation_id        - tweet_id of the root customer tweet
  turn_index             - turn number within the conversation (1, 2, 3...)
  customer_tweet_id
  customer_text
  customer_created_at
  support_tweet_id
  support_text
  support_created_at
  conversation_turns     - total turns in this conversation

Output: data/processed/amazonhelp_conversations.csv
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ──────────────────────────────────────────────
IN_PATH  = Path("data/processed/amazonhelp_raw.csv")
OUT_PATH = Path("data/processed/amazonhelp_conversations.csv")
BRAND    = "AmazonHelp"


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


if not IN_PATH.exists():
    print(f"[ERROR] Input file not found: {IN_PATH}")
    print("        Run 02_extract_amazonhelp.py first.")
    sys.exit(1)

# ════════════════════════════════════════════════════════════
# LOAD  (this file is small enough to hold fully in RAM)
# ════════════════════════════════════════════════════════════
section("LOADING amazonhelp_raw.csv")

df = pd.read_csv(IN_PATH, low_memory=False)
print(f"[OK]  Loaded {len(df):,} rows")
print(f"      Columns: {list(df.columns)}")

# Normalise in_response_to_tweet_id: float64 → nullable Int64 → int where possible
df["in_response_to_tweet_id"] = (
    pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
    .astype("Int64")        # pandas nullable integer (handles NaN)
)
df["tweet_id"] = df["tweet_id"].astype("int64")

# ════════════════════════════════════════════════════════════
# BUILD INDEXES
# ════════════════════════════════════════════════════════════
section("BUILDING INDEXES")

# tweet_id -> row (as a dict for fast lookup)
tweet_index = df.set_index("tweet_id").to_dict(orient="index")
print(f"[OK]  tweet_index built: {len(tweet_index):,} entries")

# parent_id -> list of child tweet_ids
children_of = defaultdict(list)
for _, row in df.iterrows():
    pid = row["in_response_to_tweet_id"]
    if pd.notna(pid):
        children_of[int(pid)].append(int(row["tweet_id"]))

print(f"[OK]  children_of index: {len(children_of):,} parent entries")

# Separate AmazonHelp tweets and customer tweets
amazon_mask  = df["author_id"] == BRAND
amazon_df    = df[amazon_mask].copy()
customer_df  = df[~amazon_mask].copy()

print(f"\n  AmazonHelp outbound rows : {len(amazon_df):,}")
print(f"  Customer inbound rows    : {len(customer_df):,}")

# ════════════════════════════════════════════════════════════
# FIND CONVERSATION ROOT
# We walk UP the parent chain until we reach a tweet that is:
#   a) inbound (customer tweet), AND
#   b) has no parent in our dataset (it's the conversation starter)
# We cap depth at 20 to avoid infinite loops on bad data.
# ════════════════════════════════════════════════════════════
section("FINDING CONVERSATION ROOTS")

def find_root(tweet_id, tweet_index, max_depth=20):
    """
    Walk up in_response_to_tweet_id until we reach the root tweet.
    Returns the root tweet_id, or the deepest tweet_id we can reach.
    """
    current  = tweet_id
    depth    = 0
    visited  = set()

    while depth < max_depth:
        if current in visited:
            break   # cycle guard
        visited.add(current)

        row = tweet_index.get(current)
        if row is None:
            break   # not in our dataset, current is root candidate

        pid = row.get("in_response_to_tweet_id")
        if pd.isna(pid) or pid not in tweet_index:
            break   # no parent in dataset → current is root
        current = int(pid)
        depth  += 1

    return current


# ════════════════════════════════════════════════════════════
# BUILD (CUSTOMER, AMAZON) PAIRS
# For each AmazonHelp tweet that has a known parent customer tweet,
# create one pair row.
# ════════════════════════════════════════════════════════════
section("BUILDING CONVERSATION PAIRS")

pairs        = []
skipped_no_parent   = 0
skipped_parent_missing = 0
t0 = time.time()

for i, (_, ah_row) in enumerate(amazon_df.iterrows()):
    if i % 10_000 == 0:
        print(f"  processed {i:>7,} / {len(amazon_df):,}  pairs so far: {len(pairs):,}", end="\r")

    pid = ah_row["in_response_to_tweet_id"]
    if pd.isna(pid):
        skipped_no_parent += 1
        continue

    pid = int(pid)
    parent_row = tweet_index.get(pid)
    if parent_row is None:
        # Parent tweet not in our extracted set — skip
        skipped_parent_missing += 1
        continue

    # Determine conversation root
    root_id = find_root(pid, tweet_index)

    pairs.append({
        "conversation_id"     : root_id,
        "customer_tweet_id"   : pid,
        "customer_author_id"  : parent_row.get("author_id"),
        "customer_text"       : parent_row.get("text"),
        "customer_created_at" : parent_row.get("created_at"),
        "support_tweet_id"    : int(ah_row["tweet_id"]),
        "support_text"        : ah_row["text"],
        "support_created_at"  : ah_row["created_at"],
    })

print()
print(f"\n[OK]  Pairs built         : {len(pairs):,}")
print(f"      Skipped (no parent) : {skipped_no_parent:,}")
print(f"      Skipped (parent not in dataset): {skipped_parent_missing:,}")
print(f"      Elapsed             : {time.time()-t0:.1f}s")

if not pairs:
    print("[ERROR] No pairs were built. Check the dataset structure.")
    sys.exit(1)

# ════════════════════════════════════════════════════════════
# ASSEMBLE DATAFRAME, ADD TURN INDEX & CONVERSATION LENGTH
# ════════════════════════════════════════════════════════════
section("ASSEMBLING CONVERSATION DATAFRAME")

conv_df = pd.DataFrame(pairs)

# Sort by conversation_id then by support_tweet_id (proxy for time ordering)
conv_df.sort_values(["conversation_id", "support_tweet_id"], inplace=True)
conv_df.reset_index(drop=True, inplace=True)

# Add turn_index within each conversation
conv_df["turn_index"] = conv_df.groupby("conversation_id").cumcount() + 1

# Add total turns per conversation
conv_length = conv_df.groupby("conversation_id")["turn_index"].max().rename("conversation_turns")
conv_df = conv_df.merge(conv_length, on="conversation_id", how="left")

# Reorder columns clearly
conv_df = conv_df[[
    "conversation_id",
    "turn_index",
    "conversation_turns",
    "customer_tweet_id",
    "customer_author_id",
    "customer_text",
    "customer_created_at",
    "support_tweet_id",
    "support_text",
    "support_created_at",
]]

n_conversations = conv_df["conversation_id"].nunique()
print(f"\n  Total pairs (rows)     : {len(conv_df):,}")
print(f"  Unique conversations   : {n_conversations:,}")
print(f"  Avg turns/conversation : {conv_df['conversation_turns'].mean():.2f}")
print(f"  Max turns/conversation : {conv_df['conversation_turns'].max()}")

# Show a sample conversation
sample_conv_id = conv_df[conv_df["conversation_turns"] >= 2]["conversation_id"].iloc[0] \
    if (conv_df["conversation_turns"] >= 2).any() else conv_df["conversation_id"].iloc[0]
print(f"--- Sample multi-turn conversation (id={sample_conv_id}) ---")
sample = conv_df[conv_df["conversation_id"] == sample_conv_id]
for _, r in sample.iterrows():
    print(f"  Turn {r['turn_index']}:")
    ctext = str(r['customer_text'])[:120].encode("ascii", errors="replace").decode("ascii")
    stext = str(r['support_text'])[:120].encode("ascii", errors="replace").decode("ascii")
    print(f"    Customer : {ctext}")
    print(f"    Amazon   : {stext}")

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
section("SAVING")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
conv_df.to_csv(OUT_PATH, index=False)
size_mb = OUT_PATH.stat().st_size / 1024 / 1024
print(f"[OK]  Saved: {OUT_PATH}  ({len(conv_df):,} rows, {size_mb:.2f} MB)")

section("DONE — Task 2 complete")
