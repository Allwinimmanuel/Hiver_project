import pandas as pd
from pathlib import Path
import sys
import json
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util

# Load predict_intent from api.py
sys.path.append(str(Path(__file__).parent.parent))
from api import predict_intent, clf, vectorizer

def main():
    gold_path = Path("data/evaluation/golden_set_predictions.csv")
    out_path = Path("data/evaluation/agent_predictions.csv")
    metrics_path = Path("reports/agent_evaluation.json")
    escalation_metrics_path = Path("reports/escalation_metrics.json")
    cm_path = Path("reports/confusion_matrix.png")
    
    if not gold_path.exists():
        print(f"[ERROR] Golden Set not found at {gold_path}")
        sys.exit(1)
        
    print("Loading Golden Set...")
    gold_df = pd.read_csv(gold_path)
    
    predictions = []
    confidences = []
    correctness = []
    escalation_preds = []
    
    print("Evaluating Agent against Golden Set...")
    for idx, row in gold_df.iterrows():
        true_intent = row['true_intent']
        text = str(row['customer_text'])
        
        pred, conf, needs_esc, esc_reason = predict_intent(text, clf, vectorizer)
        
        predictions.append(pred)
        confidences.append(conf)
        correctness.append(pred == true_intent)
        escalation_preds.append(1 if needs_esc else 0)
        
        if (idx+1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(gold_df)}...")

    # Create predictions df
    eval_df = pd.DataFrame({
        "example_id": gold_df["example_id"],
        "customer_text": gold_df["customer_text"],
        "true_intent": gold_df["true_intent"],
        "predicted_intent": predictions,
        "confidence": confidences,
        "correct": correctness,
        "expected_escalation": gold_df["expected_escalation"],
        "predicted_escalation": escalation_preds
    })
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(out_path, index=False)
    
    print(f"[OK] Predictions saved to {out_path}")
    
    # --- Intent Metrics ---
    y_true = eval_df["true_intent"]
    y_pred = eval_df["predicted_intent"]
    
    labels = sorted(list(set(y_true).union(set(y_pred))))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = sum(y_true == y_pred) / len(y_true)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "macro_avg": {"precision": precision_macro, "recall": recall_macro, "f1-score": f1_macro},
        "weighted_avg": {"precision": precision_weighted, "recall": recall_weighted, "f1-score": f1_weighted},
        "classification_report": report
    }
    
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Agent intent metrics saved to {metrics_path}")
    
    # --- Escalation Metrics ---
    e_true = eval_df["expected_escalation"]
    e_pred = eval_df["predicted_escalation"]
    e_p_macro, e_r_macro, e_f1_macro, _ = precision_recall_fscore_support(e_true, e_pred, average="binary", zero_division=0)
    e_accuracy = sum(e_true == e_pred) / len(e_true)
    
    esc_metrics = {
        "accuracy": e_accuracy,
        "precision": e_p_macro,
        "recall": e_r_macro,
        "f1-score": e_f1_macro
    }
    with open(escalation_metrics_path, "w") as f:
        json.dump(esc_metrics, f, indent=4)
    print(f"[OK] Escalation metrics saved to {escalation_metrics_path}")
    
    # Generate Confusion Matrix for Intent
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix: Intent Classification')
    plt.ylabel('True Intent')
    plt.xlabel('Predicted Intent')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"[OK] Confusion Matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
