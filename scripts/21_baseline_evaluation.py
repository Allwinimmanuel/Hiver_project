import pandas as pd
import json
from pathlib import Path
import sys
from sklearn.metrics import classification_report, precision_recall_fscore_support

def load_data():
    train_path = Path("data/processed/train_data.csv")
    gold_path = Path("data/evaluation/golden_set_predictions.csv")
    
    if not train_path.exists() or not gold_path.exists():
        print("[ERROR] Required data files missing.")
        sys.exit(1)
        
    train_df = pd.read_csv(train_path).dropna(subset=['intent'])
    gold_df = pd.read_csv(gold_path)
    return train_df, gold_df

def trivial_baseline(train_df, gold_df):
    majority_class = train_df['intent'].mode()[0]
    print(f"Majority class from training data: {majority_class}")
    
    y_true = gold_df['true_intent']
    y_pred = [majority_class] * len(y_true)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    acc = sum(y_true == y_pred) / len(y_true)
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return {
        "model": "Trivial Baseline (Majority Class)",
        "accuracy": acc,
        "macro_f1": f1,
        "weighted_f1": report["weighted avg"]["f1-score"],
        "classification_report": report
    }

def keyword_baseline(gold_df):
    y_true = gold_df['true_intent']
    y_pred = []
    
    for text in gold_df['customer_text']:
        text_lower = str(text).lower()
        
        if any(w in text_lower for w in ["order", "track", "status", "shipped", "when"]):
            y_pred.append("order_status_tracking")
        elif any(w in text_lower for w in ["delivery", "missing", "stolen", "lost", "arrive"]):
            y_pred.append("delivery_issue")
        elif any(w in text_lower for w in ["refund", "return", "cancel", "money back"]):
            y_pred.append("refund_return")
        elif any(w in text_lower for w in ["account", "login", "password", "charge", "billing"]):
            y_pred.append("account_billing")
        elif any(w in text_lower for w in ["prime", "membership", "subscribe"]):
            y_pred.append("prime_membership")
        elif any(w in text_lower for w in ["broken", "defect", "damage", "wrong item"]):
            y_pred.append("product_defect")
        elif any(w in text_lower for w in ["human", "agent", "real person", "escalate"]):
            y_pred.append("customer_service_escalation")
        else:
            y_pred.append("other_unknown")
            
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    acc = sum(y_true == y_pred) / len(y_true)
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return {
        "model": "Simple Baseline (Keyword Heuristics)",
        "accuracy": acc,
        "macro_f1": f1,
        "weighted_f1": report["weighted avg"]["f1-score"],
        "classification_report": report
    }

def main():
    print("=" * 60)
    print("BASELINE EVALUATION & COMPARISON")
    print("=" * 60)
    
    train_df, gold_df = load_data()
    
    # 1. Trivial Baseline
    trivial_res = trivial_baseline(train_df, gold_df)
    with open("reports/baseline_majority_results.json", "w") as f:
        json.dump(trivial_res, f, indent=4)
        
    # 2. Simple Baseline
    keyword_res = keyword_baseline(gold_df)
    with open("reports/baseline_keyword_results.json", "w") as f:
        json.dump(keyword_res, f, indent=4)
        
    # 3. Load Agent metrics
    agent_path = Path("reports/agent_evaluation.json")
    if not agent_path.exists():
        print("[ERROR] Agent evaluation not found. Run 14_evaluate_agent.py first.")
        sys.exit(1)
        
    with open(agent_path) as f:
        agent_data = json.load(f)
        
    agent_res = {
        "model": "Final Agent (TF-IDF + LR)",
        "accuracy": agent_data["accuracy"],
        "macro_f1": agent_data["macro_avg"]["f1-score"],
        "weighted_f1": agent_data["weighted_avg"]["f1-score"]
    }
    
    # 4. Compare Models
    comparison = [
        {"Model": trivial_res["model"], "Accuracy": trivial_res["accuracy"], "Macro F1": trivial_res["macro_f1"], "Weighted F1": trivial_res["weighted_f1"]},
        {"Model": keyword_res["model"], "Accuracy": keyword_res["accuracy"], "Macro F1": keyword_res["macro_f1"], "Weighted F1": keyword_res["weighted_f1"]},
        {"Model": agent_res["model"], "Accuracy": agent_res["accuracy"], "Macro F1": agent_res["macro_f1"], "Weighted F1": agent_res["weighted_f1"]}
    ]
    
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv("reports/final_model_comparison.csv", index=False)
    
    with open("reports/final_model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=4)
        
    print("\n[OK] Baselines evaluated. Comparison saved to reports/final_model_comparison.csv")
    print(comp_df.to_string(index=False))

if __name__ == "__main__":
    main()
