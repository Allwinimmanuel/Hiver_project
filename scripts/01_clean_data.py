"""
scripts/04_clean_amazonhelp.py
--------------------------------
Milestone 2, Task 3 — Data Cleaning

Loads data/processed/amazonhelp_conversations.csv and applies
conservative cleaning to remove genuinely unusable rows.

PHILOSOPHY:
  - Preserve realistic customer-support language (typos, abbreviations, slang).
  - Only remove rows that are structurally broken or completely empty.
  - Document every removal with a reason and count.
  - Never silently drop data.

Cleaning steps (in order):
  1. Remove rows where customer_text or support_text is null/empty.
  2. Remove rows where both texts are whitespace-only.
  3. Remove rows where support_text is suspiciously short (< 10 chars)
     — these are likely "DM us" fragments with no real content.
  4. Remove duplicate (customer_tweet_id, support_tweet_id) pairs.
  5. Flag (but keep by default) very long support texts (> 560 chars,
     which is unusual for Twitter).

Output: data/processed/amazonhelp_clean.csv
"""

import sys
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────
IN_PATH  = Path("data/processed/amazonhelp_conversations.csv")
OUT_PATH = Path("data/processed/amazonhelp_clean.csv")
MIN_SUPPORT_LEN = 10   # characters; below this = unusable fragment


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


if not IN_PATH.exists():
    print(f"[ERROR] Input not found: {IN_PATH}")
    print("        Run 03_reconstruct_conversations.py first.")
    sys.exit(1)

# ════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════
section("LOADING amazonhelp_conversations.csv")

df = pd.read_csv(IN_PATH, low_memory=False)
original_count = len(df)
print(f"[OK]  Loaded {original_count:,} rows")

removal_log = []   # list of (step_name, count_removed, reason)


def log_removal(step, before, after, reason):
    removed = before - after
    removal_log.append((step, removed, reason))
    if removed > 0:
        print(f"  [{step}] Removed {removed:,} rows — {reason}")
    else:
        print(f"  [{step}] 0 rows removed — {reason}")
    return after


# ════════════════════════════════════════════════════════════
# STEP 1 — Null text
# ════════════════════════════════════════════════════════════
section("STEP 1 — Removing null text")

before = len(df)
df = df[df["customer_text"].notna() & df["support_text"].notna()]
log_removal("null_text", before, len(df), "customer_text or support_text is NULL")

# ════════════════════════════════════════════════════════════
# STEP 2 — Whitespace-only text
# ════════════════════════════════════════════════════════════
section("STEP 2 — Removing whitespace-only text")

before = len(df)
df["customer_text"] = df["customer_text"].astype(str)
df["support_text"]  = df["support_text"].astype(str)

mask_whitespace = (
    (df["customer_text"].str.strip() == "") |
    (df["support_text"].str.strip()  == "")
)
df = df[~mask_whitespace]
log_removal("whitespace", before, len(df), "customer or support text is whitespace-only")

# ════════════════════════════════════════════════════════════
# STEP 3 — Too-short support responses
# ════════════════════════════════════════════════════════════
section(f"STEP 3 — Removing support responses shorter than {MIN_SUPPORT_LEN} chars")

before = len(df)
too_short_mask = df["support_text"].str.strip().str.len() < MIN_SUPPORT_LEN
# Log the actual texts being removed for transparency
short_examples = df.loc[too_short_mask, "support_text"].value_counts().head(10)
print("  Most common too-short support texts:")
for text, cnt in short_examples.items():
    safe = repr(str(text).encode("ascii", errors="replace").decode("ascii"))
    print(f"    {cnt:>5}x  {safe}")

df = df[~too_short_mask]
log_removal("too_short_support", before, len(df),
            f"support_text has fewer than {MIN_SUPPORT_LEN} characters")

# ════════════════════════════════════════════════════════════
# STEP 4 — Duplicate (customer_tweet_id, support_tweet_id) pairs
# ════════════════════════════════════════════════════════════
section("STEP 4 — Removing duplicate pairs")

before = len(df)
df.drop_duplicates(subset=["customer_tweet_id", "support_tweet_id"], inplace=True)
log_removal("duplicate_pairs", before, len(df),
            "duplicate (customer_tweet_id, support_tweet_id) pair")

# ════════════════════════════════════════════════════════════
# STEP 5 — Flag (not remove) unusually long support texts
# ════════════════════════════════════════════════════════════
section("STEP 5 — Flagging unusually long support texts (kept, not removed)")

LONG_THRESHOLD = 560  # typical Twitter limit is 280; 560 = 2x for threads
df["support_text_len"] = df["support_text"].str.len()
long_mask = df["support_text_len"] > LONG_THRESHOLD
n_long = long_mask.sum()
print(f"  Flagged {n_long:,} rows with support_text > {LONG_THRESHOLD} chars (kept)")
df["is_long_response"] = long_mask

# ════════════════════════════════════════════════════════════
# RECALCULATE CONVERSATION TURNS (some conversations shrunk)
# ════════════════════════════════════════════════════════════
section("RECALCULATING conversation turns after cleaning")

df.sort_values(["conversation_id", "support_tweet_id"], inplace=True)
df["turn_index"] = df.groupby("conversation_id").cumcount() + 1
conv_turns = df.groupby("conversation_id")["turn_index"].max().rename("conversation_turns")
df = df.drop(columns=["conversation_turns"], errors="ignore").merge(
    conv_turns, on="conversation_id", how="left"
)

df.reset_index(drop=True, inplace=True)

# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════
section("CLEANING SUMMARY")

print(f"\n  Original rows          : {original_count:,}")
print(f"  Rows after cleaning    : {len(df):,}")
print(f"  Total removed          : {original_count - len(df):,}  "
      f"({(original_count-len(df))/original_count*100:.2f}%)")
print(f"\n  Removal breakdown:")
print(f"  {'Step':<25} {'Removed':>10}  Reason")
print(f"  {'-'*65}")
for step, cnt, reason in removal_log:
    print(f"  {step:<25} {cnt:>10,}  {reason}")

print(f"\n  Unique conversations   : {df['conversation_id'].nunique():,}")
print(f"  Avg turns/conversation : {df['conversation_turns'].mean():.2f}")

# ════════════════════════════════════════════════════════════
# SHOW SAMPLE CLEAN CONVERSATIONS
# ════════════════════════════════════════════════════════════
section("SAMPLE CLEAN CONVERSATIONS")

multi_turn = df[df["conversation_turns"] >= 2]["conversation_id"].unique()
single_turn = df[df["conversation_turns"] == 1]["conversation_id"].unique()

print("\n--- Single-turn example ---")
if len(single_turn) > 0:
    ex = df[df["conversation_id"] == single_turn[0]].iloc[0]
    ctext = str(ex['customer_text'])[:200].encode("ascii", errors="replace").decode("ascii")
    stext = str(ex['support_text'])[:200].encode("ascii", errors="replace").decode("ascii")
    print(f"  Customer : {ctext}")
    print(f"  Amazon   : {stext}")

print("\n--- Multi-turn example ---")
if len(multi_turn) > 0:
    for _, row in df[df["conversation_id"] == multi_turn[0]].iterrows():
        print(f"  Turn {row['turn_index']}:")
        ctext = str(row['customer_text'])[:160].encode("ascii", errors="replace").decode("ascii")
        stext = str(row['support_text'])[:160].encode("ascii", errors="replace").decode("ascii")
        print(f"    Customer : {ctext}")
        print(f"    Amazon   : {stext}")

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
section("SAVING")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
size_mb = OUT_PATH.stat().st_size / 1024 / 1024
print(f"[OK]  Saved: {OUT_PATH}  ({len(df):,} rows, {size_mb:.2f} MB)")

section("DONE — Task 3 complete")
