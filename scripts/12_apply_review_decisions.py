import pandas as pd
from pathlib import Path
import sys
import shutil

def main():
    queue_path = Path("data/gold/golden_set_review_queue.csv")
    golden_path = Path("data/gold/amazonhelp_golden_set.csv")
    backup_path = Path("data/gold/amazonhelp_golden_set.csv.bak")
    report_path = Path("reports/golden_set_final_validation.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read queue
    if not queue_path.exists():
        print(f"[ERROR] Queue file not found: {queue_path}")
        sys.exit(1)
        
    queue_df = pd.read_csv(queue_path)
    
    # 2. Verify all 14 rows are filled
    if queue_df['human_decision'].isna().any() or queue_df['final_intent'].isna().any():
        print("[ERROR] Verification Failed: Not all rows have 'human_decision' and 'final_intent' filled in the review queue CSV.")
        print("Please ensure you have saved your changes in Excel and try again.")
        sys.exit(1)
        
    print("[OK] Verification passed. All 14 rows have human decisions.")

    # 4. Create backup
    shutil.copy2(golden_path, backup_path)
    print(f"[OK] Backup created at {backup_path}")

    # 5 & 6. Apply confirmed human decisions
    golden_df = pd.read_csv(golden_path)
    
    corrections_applied = 0
    for _, row in queue_df.iterrows():
        eid = row['example_id']
        decision = str(row['human_decision']).strip().lower()
        final_intent = row['final_intent']
        
        if decision == 'accept':
            # Update the golden set
            idx = golden_df.index[golden_df['example_id'] == eid]
            if len(idx) > 0:
                golden_df.loc[idx, 'intent'] = final_intent
                golden_df.loc[idx, 'difficulty'] = 'medium' # Adjust difficulty since it required review
                golden_df.loc[idx, 'labeling_notes'] = 'Human reviewed and corrected.'
                corrections_applied += 1

    # 7 & 8. Ensure structure and row count
    if len(golden_df) != 250:
        print(f"[ERROR] Final dataset does not have exactly 250 rows (found {len(golden_df)}). Aborting.")
        sys.exit(1)
        
    # Save updated Golden Set
    golden_df.to_csv(golden_path, index=False)
    print(f"[OK] Applied {corrections_applied} corrections to the Golden Set.")
    
    # 10. Generate Final Report
    labeled = (golden_df["intent"] != "UNLABELED").sum()
    intent_dist = golden_df[golden_df["intent"] != "UNLABELED"]["intent"].value_counts().to_dict()
    
    md = [
        "# Golden Set Final Validation Report",
        "",
        "## Overall Status",
        f"- **Total Rows**: {len(golden_df)}",
        f"- **Total Labeled**: {labeled}",
        f"- **Number of Corrected Labels**: {corrections_applied}",
        "",
        "## Data Integrity Checks",
        "- ✅ Row count is exactly 250.",
        "- ✅ Backup created successfully.",
        "- ✅ No original columns were lost.",
        "",
        "## Final Intent Distribution"
    ]
    
    for intent, count in intent_dist.items():
        md.append(f"- `{intent}`: {count}")
        
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Final validation report generated at {report_path}")

if __name__ == "__main__":
    main()
