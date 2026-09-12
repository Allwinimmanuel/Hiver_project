import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def main():
    in_path = Path("data/processed/amazonhelp_weak_labeled.csv")
    out_train_path = Path("data/processed/train_data.csv")
    out_val_path = Path("data/processed/val_data.csv")
    report_path = Path("reports/training_data_summary.md")
    
    df = pd.read_csv(in_path)
    total_weak = len(df)
    
    # 2. Remove rows with missing customer_text
    df = df.dropna(subset=['customer_text'])
    
    # 3. Remove extremely low-confidence examples
    # Since confidence is either 0.9, 0.8, 0.6, let's keep all for now to have enough data,
    # or just remove if we decided a specific threshold. We'll keep >= 0.6.
    df = df[df['confidence'] >= 0.5]
    
    # We could optionally undersample "other_unknown" to avoid severe class imbalance
    other_df = df[df['intent'] == 'other_unknown']
    known_df = df[df['intent'] != 'other_unknown']
    
    # Undersample other_unknown to the size of the known_df so it doesn't dominate completely
    # (or up to a max limit)
    if len(other_df) > len(known_df):
        other_df = other_df.sample(n=len(known_df), random_state=42)
        
    df = pd.concat([known_df, other_df]).reset_index(drop=True)
    
    retained = len(df)
    removed = total_weak - retained
    
    class_dist = df['intent'].value_counts().to_dict()
    
    # 6. Split data
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['intent'], random_state=42)
    
    train_df.to_csv(out_train_path, index=False)
    val_df.to_csv(out_val_path, index=False)
    
    md = [
        "# Training Data Summary",
        "",
        f"- **Total weakly labeled examples**: {total_weak}",
        f"- **Number retained for training/validation**: {retained}",
        f"- **Number removed (missing text, low confidence, or undersampled noise)**: {removed}",
        "",
        "## Class Distribution",
    ]
    
    for intent, count in class_dist.items():
        md.append(f"- `{intent}`: {count}")
        
    md.extend([
        "",
        "## Split Sizes",
        f"- **Training Set**: {len(train_df)}",
        f"- **Validation Set**: {len(val_df)}",
        "",
        "## Limitations of Weak Supervision",
        "The intents assigned here are generated from deterministic keywords (e.g., regex matching 'missing package'). "
        "As we saw in the Golden Set audit, this approach is extremely rigid and will incorrectly classify context-heavy or linguistically "
        "diverse messages as `other_unknown`. A model trained on this dataset will likely learn to mimic these flawed rules rather than "
        "true human reasoning, artificially inflating its performance on similar regex-labeled data while struggling on human-verified ground truth."
    ])
    
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Training data prepared. Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"[OK] Report saved to {report_path}")

if __name__ == "__main__":
    main()
