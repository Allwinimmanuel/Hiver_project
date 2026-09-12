"""
scripts/05_analyze_amazonhelp.py
---------------------------------
Milestone 2, Tasks 4 & 5 — Conversation Analysis + Data Quality Review

Loads:  data/processed/amazonhelp_clean.csv
        data/processed/amazonhelp_conversations.csv  (pre-clean, for comparison)

Produces:
  reports/amazonhelp_data_analysis.md
  reports/amazonhelp_conversation_stats.csv
"""

import sys
import re
from pathlib import Path
from collections import Counter

import pandas as pd

MIN_SUPPORT_LEN = 10   # must match the value used in 04_clean_amazonhelp.py

# ──────────────────────────────────────────────
CLEAN_PATH  = Path("data/processed/amazonhelp_clean.csv")
RAW_CONV    = Path("data/processed/amazonhelp_conversations.csv")
RAW_EXTRACT = Path("data/processed/amazonhelp_raw.csv")
REPORTS_DIR = Path("reports")


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


for p in [CLEAN_PATH, RAW_CONV, RAW_EXTRACT]:
    if not p.exists():
        print(f"[ERROR] Required file missing: {p}")
        sys.exit(1)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════
# LOAD
# ════════════════════════════════════════════════════════════
section("LOADING DATA")

clean   = pd.read_csv(CLEAN_PATH,  low_memory=False)
raw_conv= pd.read_csv(RAW_CONV,    low_memory=False)
raw_ext = pd.read_csv(RAW_EXTRACT, low_memory=False)

print(f"[OK]  amazonhelp_raw.csv           : {len(raw_ext):,} rows")
print(f"[OK]  amazonhelp_conversations.csv : {len(raw_conv):,} rows")
print(f"[OK]  amazonhelp_clean.csv         : {len(clean):,} rows")

# ════════════════════════════════════════════════════════════
# BASIC COUNTS
# ════════════════════════════════════════════════════════════
section("BASIC COUNTS")

total_extracted       = len(raw_ext)
total_outbound_raw    = (raw_ext["author_id"] == "AmazonHelp").sum()
total_inbound_raw     = (raw_ext["author_id"] != "AmazonHelp").sum()
total_pairs_preClean  = len(raw_conv)
total_pairs_clean     = len(clean)
n_conv_preClean       = raw_conv["conversation_id"].nunique()
n_conv_clean          = clean["conversation_id"].nunique()
pairs_removed         = total_pairs_preClean - total_pairs_clean
pairs_removed_pct     = pairs_removed / total_pairs_preClean * 100 if total_pairs_preClean else 0

print(f"  Total extracted tweets (raw)    : {total_extracted:,}")
print(f"    AmazonHelp outbound           : {total_outbound_raw:,}")
print(f"    Customer inbound              : {total_inbound_raw:,}")
print(f"  Pairs before cleaning           : {total_pairs_preClean:,}")
print(f"  Pairs after cleaning            : {total_pairs_clean:,}")
print(f"  Pairs removed                   : {pairs_removed:,}  ({pairs_removed_pct:.2f}%)")
print(f"  Conversations before cleaning   : {n_conv_preClean:,}")
print(f"  Conversations after cleaning    : {n_conv_clean:,}")

# ════════════════════════════════════════════════════════════
# CONVERSATION LENGTH DISTRIBUTION
# ════════════════════════════════════════════════════════════
section("CONVERSATION LENGTH DISTRIBUTION")

# One row per conversation (take max turn as length)
conv_lengths = (
    clean.groupby("conversation_id")["conversation_turns"].max()
)

avg_turns    = conv_lengths.mean()
median_turns = conv_lengths.median()
max_turns    = conv_lengths.max()
min_turns    = conv_lengths.min()
single_turn  = (conv_lengths == 1).sum()
multi_turn   = (conv_lengths >= 2).sum()

print(f"  Average turns per conversation  : {avg_turns:.2f}")
print(f"  Median  turns per conversation  : {median_turns:.1f}")
print(f"  Min turns                       : {int(min_turns)}")
print(f"  Max turns                       : {int(max_turns)}")
print(f"  Single-turn conversations       : {single_turn:,}  ({single_turn/n_conv_clean*100:.1f}%)")
print(f"  Multi-turn  conversations       : {multi_turn:,}  ({multi_turn/n_conv_clean*100:.1f}%)")

# Distribution table
print("\n  Turn-length distribution:")
dist = conv_lengths.value_counts().sort_index()
for turns, count in dist.items():
    bar = "#" * min(int(count / dist.max() * 30), 30)
    print(f"    {int(turns):>3} turn(s) : {count:>7,}  {bar}")

# ════════════════════════════════════════════════════════════
# TEXT LENGTH STATS
# ════════════════════════════════════════════════════════════
section("TEXT LENGTH STATISTICS")

clean["customer_len"] = clean["customer_text"].str.len()
clean["support_len"]  = clean["support_text"].str.len()

print("  Customer message lengths:")
print(f"    Mean   : {clean['customer_len'].mean():.1f} chars")
print(f"    Median : {clean['customer_len'].median():.1f} chars")
print(f"    Max    : {clean['customer_len'].max()} chars")

print("  Support message lengths:")
print(f"    Mean   : {clean['support_len'].mean():.1f} chars")
print(f"    Median : {clean['support_len'].median():.1f} chars")
print(f"    Max    : {clean['support_len'].max()} chars")

# ════════════════════════════════════════════════════════════
# DATE RANGE
# ════════════════════════════════════════════════════════════
section("DATE RANGE")

try:
    clean["customer_created_at"] = pd.to_datetime(clean["customer_created_at"], errors="coerce")
    clean["support_created_at"]  = pd.to_datetime(clean["support_created_at"],  errors="coerce")
    earliest = clean["customer_created_at"].min()
    latest   = clean["customer_created_at"].max()
    print(f"  Earliest conversation : {earliest}")
    print(f"  Latest  conversation  : {latest}")
    date_range_str = f"{earliest} → {latest}"
except Exception as e:
    print(f"  [WARN] Could not parse dates: {e}")
    date_range_str = "N/A"

# ════════════════════════════════════════════════════════════
# TOP RECURRING WORDS IN CUSTOMER MESSAGES
# ════════════════════════════════════════════════════════════
section("TOP RECURRING WORDS IN CUSTOMER MESSAGES (support-topic signal)")

STOP_WORDS = {
    "i", "me", "my", "the", "a", "an", "and", "or", "but", "to", "of",
    "in", "is", "it", "you", "your", "we", "our", "this", "that", "for",
    "on", "at", "with", "have", "has", "be", "am", "are", "was", "were",
    "been", "do", "did", "does", "not", "no", "so", "if", "can", "will",
    "would", "could", "should", "from", "by", "as", "about", "up", "out",
    "s", "t", "re", "ve", "ll", "just", "get", "got", "im", "its", "hi",
    "hey", "please", "amp", "amazonhelp", "amazon", "help", "need", "want",
    "how", "what", "when", "why", "where", "who", "any", "all", "some",
    "more", "still", "also", "he", "she", "they", "them", "their",
}

word_counter = Counter()
# Sample up to 50k rows for speed
sample_texts = clean["customer_text"].dropna().sample(
    min(50_000, len(clean)), random_state=42
)
for text in sample_texts:
    words = re.findall(r"[a-z]+", str(text).lower())
    word_counter.update(w for w in words if w not in STOP_WORDS and len(w) > 2)

print("  Top 25 words in customer messages:")
for word, cnt in word_counter.most_common(25):
    print(f"    {word:<20} {cnt:>7,}")

# ════════════════════════════════════════════════════════════
# DATA QUALITY ISSUES (Task 5)
# ════════════════════════════════════════════════════════════
section("DATA QUALITY — Detailed Issues")

# Orphan check: AmazonHelp tweets with no parent in dataset
ah_raw = raw_ext[raw_ext["author_id"] == "AmazonHelp"].copy()
ah_raw["in_response_to_tweet_id"] = pd.to_numeric(
    ah_raw["in_response_to_tweet_id"], errors="coerce"
).astype("Int64")
known_ids = set(raw_ext["tweet_id"].astype(int).tolist())
orphan_mask = ah_raw["in_response_to_tweet_id"].notna() & \
              ~ah_raw["in_response_to_tweet_id"].astype("Int64").isin(known_ids)
n_orphan = orphan_mask.sum()
n_no_parent = ah_raw["in_response_to_tweet_id"].isna().sum()

# Extremely long conversations
long_convs = conv_lengths[conv_lengths >= 10]

# Very short customer messages (< 5 chars — almost certainly junk)
very_short_customer = (clean["customer_len"] < 5).sum()

print(f"  Orphan AH tweets (parent tweet_id present but not in dataset): {n_orphan:,}")
print(f"  AH tweets with no in_response_to_tweet_id at all            : {n_no_parent:,}")
print(f"  Conversations with >= 10 turns (extremely long)             : {len(long_convs):,}")
print(f"  Clean pairs with customer_text < 5 chars                    : {very_short_customer:,}")

if len(long_convs) > 0:
    print(f"\n  Longest conversation IDs:")
    for cid, turns in long_convs.sort_values(ascending=False).head(5).items():
        print(f"    conversation_id={cid}  turns={turns}")

# ════════════════════════════════════════════════════════════
# MISSING TEXT % in cleaned dataset
# ════════════════════════════════════════════════════════════
cust_null_pct = clean["customer_text"].isna().mean() * 100
supp_null_pct = clean["support_text"].isna().mean() * 100
print(f"\n  Null customer_text in clean set : {cust_null_pct:.2f}%")
print(f"  Null support_text  in clean set : {supp_null_pct:.2f}%")

# ════════════════════════════════════════════════════════════
# SAMPLE CONVERSATIONS FOR REPORT
# ════════════════════════════════════════════════════════════
section("COLLECTING SAMPLE CONVERSATIONS")

examples = []
for cid in clean[clean["conversation_turns"] == 1]["conversation_id"].unique()[:2]:
    rows = clean[clean["conversation_id"] == cid]
    examples.append(("Single-turn", rows))

for cid in clean[clean["conversation_turns"] >= 2]["conversation_id"].unique()[:2]:
    rows = clean[clean["conversation_id"] == cid]
    examples.append(("Multi-turn", rows))

# ════════════════════════════════════════════════════════════
# WRITE amazonhelp_conversation_stats.csv
# ════════════════════════════════════════════════════════════
section("WRITING reports/amazonhelp_conversation_stats.csv")

# One row per conversation
conv_stats = (
    clean.groupby("conversation_id")
    .agg(
        n_turns          =("turn_index", "max"),
        first_customer_text=("customer_text", "first"),
        first_support_text =("support_text",  "first"),
        total_customer_chars=("customer_len", "sum"),
        total_support_chars =("support_len",  "sum"),
    )
    .reset_index()
)
stats_path = REPORTS_DIR / "amazonhelp_conversation_stats.csv"
conv_stats.to_csv(stats_path, index=False)
print(f"[OK]  Saved: {stats_path}  ({len(conv_stats):,} conversations)")

# ════════════════════════════════════════════════════════════
# BUILD EXAMPLE MARKDOWN BLOCKS
# ════════════════════════════════════════════════════════════
def make_example_block(label, rows):
    lines = [f"**{label}** (conversation_id = {rows.iloc[0]['conversation_id']})"]
    lines.append("")
    for _, r in rows.iterrows():
        lines.append(f"- **Turn {int(r['turn_index'])}**")
        ctext = str(r['customer_text']).replace("\n", " ").strip()[:300]
        stext = str(r['support_text']).replace("\n", " ").strip()[:300]
        lines.append(f"  - 🙋 Customer: {ctext}")
        lines.append(f"  - 🤖 AmazonHelp: {stext}")
    return "\n".join(lines)

example_blocks = "\n\n---\n\n".join(
    make_example_block(label, rows) for label, rows in examples
)

# Length distribution markdown
dist_rows = []
for turns, count in dist.items():
    dist_rows.append(f"| {int(turns)} | {count:,} | {count/n_conv_clean*100:.1f}% |")
dist_md = (
    "| Turns | Conversations | % |\n|-------|--------------|---|\n"
    + "\n".join(dist_rows)
)

# ════════════════════════════════════════════════════════════
# WRITE amazonhelp_data_analysis.md
# ════════════════════════════════════════════════════════════
section("WRITING reports/amazonhelp_data_analysis.md")

report = f"""# AmazonHelp Data Analysis Report
Generated by: `scripts/05_analyze_amazonhelp.py`

---

## 1. Why AmazonHelp Was Selected

AmazonHelp was selected from 108 support brands in the TWCS dataset because:
- **Largest corpus**: 169,840 outbound (brand → customer) messages — nearly 60% more than the second-place AppleSupport (106,860).
- **Domain diversity**: Amazon handles orders, shipping, refunds, account issues, and product questions — broad enough to make a general-purpose support agent.
- **Clean data**: No major data quality flags were raised in Milestone 1 inspection.
- **Recognisable brand**: Evaluators immediately understand the context of every Q&A pair.

---

## 2. Extraction Statistics

| Metric | Value |
|--------|-------|
| Total extracted rows (raw) | {total_extracted:,} |
| AmazonHelp outbound rows | {total_outbound_raw:,} |
| Customer inbound rows | {total_inbound_raw:,} |

---

## 3. Conversation Reconstruction

| Metric | Value |
|--------|-------|
| (Customer, Amazon) pairs before cleaning | {total_pairs_preClean:,} |
| Unique conversations before cleaning | {n_conv_preClean:,} |
| Pairs removed during cleaning | {pairs_removed:,} ({pairs_removed_pct:.2f}%) |
| **Valid pairs after cleaning** | **{total_pairs_clean:,}** |
| **Valid conversations after cleaning** | **{n_conv_clean:,}** |

---

## 4. Conversation Statistics

| Metric | Value |
|--------|-------|
| Average turns per conversation | {avg_turns:.2f} |
| Median turns per conversation | {median_turns:.1f} |
| Min turns | {int(min_turns)} |
| Max turns | {int(max_turns)} |
| Single-turn conversations | {single_turn:,} ({single_turn/n_conv_clean*100:.1f}%) |
| Multi-turn conversations | {multi_turn:,} ({multi_turn/n_conv_clean*100:.1f}%) |

### Turn-Length Distribution

{dist_md}

### Message Length Statistics

| Direction | Mean (chars) | Median (chars) | Max (chars) |
|-----------|-------------|----------------|-------------|
| Customer messages | {clean['customer_len'].mean():.1f} | {clean['customer_len'].median():.1f} | {int(clean['customer_len'].max())} |
| Support messages | {clean['support_len'].mean():.1f} | {clean['support_len'].median():.1f} | {int(clean['support_len'].max())} |

---

## 5. Data Cleaning Statistics

| Step | Rows Removed | Reason |
|------|-------------|--------|
| Null text | see script output | customer_text or support_text is NULL |
| Whitespace only | see script output | Text contains only whitespace |
| Too-short support (< {MIN_SUPPORT_LEN} chars) | see script output | Support text too short to be useful |
| Duplicate pairs | see script output | Duplicate (customer_tweet_id, support_tweet_id) |
| **Total removed** | **{pairs_removed:,}** | **{pairs_removed_pct:.2f}% of pre-clean pairs** |

> Note: Run `scripts/04_clean_amazonhelp.py` to see exact removal counts per step.

---

## 6. Data Quality Issues

| Issue | Count | Action |
|-------|-------|--------|
| Orphan AH tweets (parent missing from dataset) | {n_orphan:,} | Excluded from pairs (no customer text to pair) |
| AH tweets with no in_response_to_tweet_id | {n_no_parent:,} | Excluded (cannot determine conversation) |
| Conversations with ≥ 10 turns | {len(long_convs):,} | Kept (may be complex support threads) |
| Clean pairs with customer_text < 5 chars | {very_short_customer:,} | Kept (may be valid one-word queries) |
| Null customer_text in clean set | {cust_null_pct:.2f}% | N/A (none after cleaning) |
| Null support_text in clean set | {supp_null_pct:.2f}% | N/A (none after cleaning) |

**Date range of conversations:** {date_range_str}

---

## 7. Top Customer Query Topics

Top 25 words from a sample of 50,000 customer messages (stop-words removed):

| Word | Count | Word | Count |
|------|-------|------|-------|
{chr(10).join(
    f"| {r[0][0]} | {r[0][1]:,} | {r[1][0]} | {r[1][1]:,} |"
    for r in zip(
        word_counter.most_common(25)[0::2],
        word_counter.most_common(25)[1::2],
    )
)}

---

## 8. Example Reconstructed Conversations

{example_blocks}

---

## 9. Suitability Recommendation

**AmazonHelp is highly suitable for the AI agent. ✅**

| Criteria | Assessment |
|----------|-----------|
| Corpus size | ✅ {total_pairs_clean:,} clean (customer, support) pairs — large enough for retrieval-based and fine-tuning approaches |
| Conversation quality | ✅ {multi_turn:,} multi-turn conversations provide rich context |
| Topic diversity | ✅ Orders, shipping, refunds, account — broad coverage |
| Data cleanliness | ✅ Only {pairs_removed_pct:.2f}% of pairs removed during cleaning |
| Reproducibility | ✅ All tweet IDs preserved; extraction is deterministic |

> **Next step (Milestone 3):** Define intent categories from the customer messages,
> and build the retrieval/matching pipeline for the AI support agent.
"""

md_path = REPORTS_DIR / "amazonhelp_data_analysis.md"
md_path.write_text(report, encoding="utf-8")
print(f"[OK]  Saved: {md_path}")

section("DONE — Tasks 4 & 5 complete")
print(f"\nSummary:")
print(f"  Valid (customer, support) pairs : {total_pairs_clean:,}")
print(f"  Valid conversations             : {n_conv_clean:,}")
print(f"  Average turns / conversation    : {avg_turns:.2f}")
print(f"\nReady for Milestone 3.")
