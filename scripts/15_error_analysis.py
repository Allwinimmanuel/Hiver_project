import pandas as pd
from pathlib import Path
import json

def main():
    pred_path = Path("data/evaluation/golden_set_predictions.csv")
    report_path = Path("reports/milestone4_error_analysis.md")
    
    if not pred_path.exists():
        print(f"[ERROR] Predictions not found at {pred_path}")
        return
        
    df = pd.read_csv(pred_path)
    
    # 1. Incorrect predictions
    errors = df[~df['correct']]
    
    # 2 & 3. Error counts
    errors_by_true = errors['true_intent'].value_counts()
    errors_by_pred = errors['predicted_intent'].value_counts()
    
    # Common confusion pairs
    pairs = errors.groupby(['true_intent', 'predicted_intent']).size().reset_index(name='count')
    pairs = pairs.sort_values('count', ascending=False)
    
    md = [
        "# Milestone 4 Error Analysis",
        "",
        f"**Total Errors**: {len(errors)} / {len(df)}",
        "",
        "## Top Confusion Pairs",
        "| True Intent | Predicted Intent | Count |",
        "|---|---|---|"
    ]
    for _, row in pairs.head(10).iterrows():
        md.append(f"| {row['true_intent']} | {row['predicted_intent']} | {row['count']} |")
        
    md.extend([
        "",
        "## Errors by True Intent (Recall Failures)",
    ])
    for intent, count in errors_by_true.items():
        md.append(f"- `{intent}`: {count}")
        
    md.extend([
        "",
        "## Errors by Predicted Intent (Precision Failures)",
    ])
    for intent, count in errors_by_pred.items():
        md.append(f"- `{intent}`: {count}")
        
    # Over-prediction of other_unknown
    other_pred_count = errors_by_pred.get('other_unknown', 0)
    md.extend([
        "",
        "## Specific Problem Areas",
        f"- **Over-prediction of `other_unknown`**: The model incorrectly guessed `other_unknown` {other_pred_count} times. This is likely because the weak supervision training data heavily skewed towards `other_unknown` (making it the default \"safe\" guess), combined with noisy Twitter language that doesn't strongly match any TF-IDF vocabulary for specific intents.",
        "",
        "## Sample Misclassifications",
    ])
    
    # Sample 15 misclassifications
    sample = errors.head(15)
    for idx, row in sample.iterrows():
        md.extend([
            f"### Example {row['example_id']}",
            f"- **Customer Text**: {row['customer_text']}",
            f"- **True Intent**: `{row['true_intent']}`",
            f"- **Predicted Intent**: `{row['predicted_intent']}` (Conf: {row['confidence']})",
            ""
        ])
        
    md.extend([
        "## Likely Reasons for Failure",
        "- **Weak Supervision Noise**: The model learned the flawed regex rules from the training data rather than true intent.",
        "- **Noisy Twitter Language**: Abbreviations, typos, and fragmented sentences dilute TF-IDF signals.",
        "- **Class Imbalance**: Despite undersampling, the overwhelming majority of Twitter data is `other_unknown` noise.",
        "- **Multilingual Messages**: Non-English texts confuse the English-centric TF-IDF vocabulary.",
        "- **Context Dependence**: Short tweets (e.g. 'Yes thanks') lack the semantic keywords needed by a bag-of-words model."
    ])
        
    # Generate Failure Analysis CSV as required by final checklist
    failure_records = []
    for idx, row in errors.iterrows():
        failure_cat = "Wrong Classification"
        reason = "TF-IDF Vocabulary Mismatch or Missing Context"
        suggestion = "Switch to Sentence-Transformers (Embeddings)"
        
        # Specific heuristic reasons
        if row['predicted_intent'] == 'other_unknown':
            reason = "Defaulted to majority class due to weak signal"
            suggestion = "Increase class weight or use LLM router"
        elif row['confidence'] < 0.6:
            failure_cat = "Low Confidence Misclassification"
            
        failure_records.append({
            "Input Ticket": row['customer_text'],
            "Expected Output (Category)": row['true_intent'],
            "Actual Output (Category)": row['predicted_intent'],
            "Failure Category": failure_cat,
            "Possible Reason": reason,
            "Suggested Improvement": suggestion
        })
        
    fail_csv_path = Path("reports/failure_analysis.csv")
    fail_df = pd.DataFrame(failure_records)
    fail_df.to_csv(fail_csv_path, index=False)
    print(f"[OK] Failure analysis CSV saved to {fail_csv_path}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Error analysis report saved to {report_path}")

if __name__ == "__main__":
    main()
