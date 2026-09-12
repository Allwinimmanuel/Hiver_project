import pandas as pd
from pathlib import Path
import sys
import json
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util

# Load predict_intent dynamically
spec = importlib.util.spec_from_file_location("classifier", "scripts/13_agent_classifier.py")
classifier = importlib.util.module_from_spec(spec)
sys.modules["classifier"] = classifier
spec.loader.exec_module(classifier)
predict_intent = classifier.predict_intent

def main():
    gold_path = Path("data/gold/amazonhelp_golden_set.csv")
    out_path = Path("data/evaluation/golden_set_predictions.csv")
    metrics_path = Path("reports/milestone4_metrics.json")
    cm_path = Path("reports/confusion_matrix.png")
    
    if not gold_path.exists():
        print(f"[ERROR] Golden Set not found at {gold_path}")
        sys.exit(1)
        
    print("Loading Golden Set...")
    gold_df = pd.read_csv(gold_path)
    
    if len(gold_df) != 250:
        print(f"[ERROR] Golden Set must have exactly 250 rows (found {len(gold_df)})")
        sys.exit(1)
        
    predictions = []
    confidences = []
    correctness = []
    
    print("Evaluating Agent against Golden Set...")
    for idx, row in gold_df.iterrows():
        true_intent = row['intent']
        text = str(row['customer_text'])
        
        pred, conf = predict_intent(text)
        
        predictions.append(pred)
        confidences.append(conf)
        correctness.append(pred == true_intent)
        
        if (idx+1) % 50 == 0:
            print(f"  Processed {idx+1}/250...")

    # Create predictions df
    eval_df = pd.DataFrame({
        "example_id": gold_df["example_id"],
        "customer_text": gold_df["customer_text"],
        "true_intent": gold_df["intent"],
        "predicted_intent": predictions,
        "confidence": confidences,
        "correct": correctness
    })
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(out_path, index=False)
    
    print(f"[OK] Predictions saved to {out_path}")
    
    # Calculate metrics
    y_true = eval_df["true_intent"]
    y_pred = eval_df["predicted_intent"]
    
    labels = sorted(list(set(y_true).union(set(y_pred))))
    
    # P, R, F1
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    # Accuracy
    accuracy = sum(y_true == y_pred) / len(y_true)
    
    # Report dict
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "macro_avg": {
            "precision": precision_macro,
            "recall": recall_macro,
            "f1-score": f1_macro
        },
        "weighted_avg": {
            "precision": precision_weighted,
            "recall": recall_weighted,
            "f1-score": f1_weighted
        },
        "classification_report": report
    }
    
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"[OK] Metrics saved to {metrics_path}")
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix: Golden Set vs Agent Predictions')
    plt.ylabel('True Intent')
    plt.xlabel('Predicted Intent')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    
    print(f"[OK] Confusion Matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
