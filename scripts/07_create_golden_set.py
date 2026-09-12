"""
scripts/07_create_golden_set.py
-------------------------------
Milestone 3, Tasks 4 & 5 — Golden Set Sampling Strategy

This script samples 250 diverse customer messages from the cleaned
dataset to create a Golden Set for manual labeling.

Sampling Strategy:
1. Load data/processed/amazonhelp_clean.csv
2. Keep only the first turn of each conversation (where the issue is usually stated).
3. Bin by text length (short, medium, long) to ensure varied complexity.
4. Take a stratified random sample across these length bins.
5. Generate the output CSV with placeholder columns.
"""


from pathlib import Path
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
IN_PATH = Path("data/processed/amazonhelp_clean.csv")
OUT_PATH = Path("data/gold/amazonhelp_golden_set.csv")
SAMPLE_SIZE = 250

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

if not IN_PATH.exists():
    print(f"[ERROR] Clean dataset not found: {IN_PATH}")
    sys.exit(1)

section("LOADING DATA & FILTERING")
df = pd.read_csv(IN_PATH, low_memory=False)
print(f"Total rows loaded: {len(df):,}")

# Keep only Turn 1 (the initial customer message)
df_root = df[df["turn_index"] == 1].copy()
print(f"Total first-turn messages: {len(df_root):,}")

# Add length bins
df_root["text_len"] = df_root["customer_text"].str.len()
df_root["len_bin"] = pd.qcut(df_root["text_len"], q=3, labels=["short", "medium", "long"])

section("SAMPLING")
# Stratified sampling based on length bins to ensure we don't only get short or only long messages
np.random.seed(42)  # For reproducibility

samples_per_bin = SAMPLE_SIZE // 3
sampled_dfs = []

for bin_name in ["short", "medium", "long"]:
    bin_data = df_root[df_root["len_bin"] == bin_name]
    # Sample without replacement
    sampled = bin_data.sample(n=min(samples_per_bin, len(bin_data)), random_state=42)
    sampled_dfs.append(sampled)

# If we need a few more to reach exactly 250 (due to rounding)
golden_df = pd.concat(sampled_dfs)
if len(golden_df) < SAMPLE_SIZE:
    remainder = SAMPLE_SIZE - len(golden_df)
    remaining_data = df_root[~df_root.index.isin(golden_df.index)]
    extra = remaining_data.sample(n=remainder, random_state=42)
    golden_df = pd.concat([golden_df, extra])

# Shuffle the final set
golden_df = golden_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Generate example_id
golden_df["example_id"] = [f"GS-{i:03d}" for i in range(1, len(golden_df) + 1)]

# Add annotation columns
golden_df["intent"] = "UNLABELED"
golden_df["labeling_notes"] = ""
golden_df["difficulty"] = ""
golden_df["source_conversation_id"] = golden_df["conversation_id"]

# Select final columns
final_cols = [
    "example_id",
    "customer_tweet_id",
    "customer_text",
    "intent",
    "labeling_notes",
    "source_conversation_id",
    "difficulty",
    "text_len" # Keeping for reference during labeling
]

golden_export = golden_df[final_cols]

section("SAVING GOLDEN SET")
golden_export.to_csv(OUT_PATH, index=False)
print(f"[OK] Generated {len(golden_export)} examples.")
print(f"[OK] Saved Golden Set to {OUT_PATH}")
print(f"Length distribution in sample:")
print(golden_df["len_bin"].value_counts())
