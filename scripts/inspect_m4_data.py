import pandas as pd
from pathlib import Path

def main():
    clean_path = Path("data/processed/amazonhelp_clean.csv")
    gold_path = Path("data/gold/amazonhelp_golden_set.csv")
    report_path = Path("reports/milestone4_data_inspection.md")
    
    clean_df = pd.read_csv(clean_path)
    gold_df = pd.read_csv(gold_path)
    
    clean_rows = len(clean_df)
    clean_cols = list(clean_df.columns)
    clean_missing = clean_df.isna().sum().to_dict()
    has_labels = 'intent' in clean_cols
    
    gold_rows = len(gold_df)
    gold_cols = list(gold_df.columns)
    gold_missing = gold_df.isna().sum().to_dict()
    
    md = [
        "# Milestone 4 Data Inspection Report",
        "",
        "## Cleaned Dataset (`data/processed/amazonhelp_clean.csv`)",
        f"- **Rows**: {clean_rows}",
        f"- **Columns**: {', '.join(clean_cols)}",
        "- **Missing Values**:",
    ]
    for col, count in clean_missing.items():
        if count > 0:
            md.append(f"  - `{col}`: {count} missing")
    if not any(v > 0 for v in clean_missing.values()):
        md.append("  - None")
        
    md.extend([
        f"- **Intent Labels Exist?**: {'Yes' if has_labels else 'No'}",
        "",
        "## Golden Set (`data/gold/amazonhelp_golden_set.csv`)",
        f"- **Rows**: {gold_rows}",
        f"- **Columns**: {', '.join(gold_cols)}",
        "- **Missing Values**:",
    ])
    for col, count in gold_missing.items():
        if count > 0:
            md.append(f"  - `{col}`: {count} missing")
    if not any(v > 0 for v in gold_missing.values()):
        md.append("  - None")
        
    md.extend([
        "",
        "## Recommended Training Strategy",
        "Since supervised intent labels do not exist in the cleaned dataset, we must employ a **weak-supervision labeling pipeline**. We will write deterministic keyword and phrase-based rules to assign intents (and confidence scores) to the `customer_text` in `amazonhelp_clean.csv`. This weakly-labeled data will then serve as the training set for our Logistic Regression classifier."
    ])
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Generated {report_path}")

if __name__ == "__main__":
    main()
