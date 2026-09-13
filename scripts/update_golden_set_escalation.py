import pandas as pd
from pathlib import Path

def add_expected_escalation():
    gold_path = Path("data/evaluation/golden_set_predictions.csv")
    df = pd.read_csv(gold_path)
    
    # Calculate expected escalation based on true_intent and text heuristics
    def calculate_expected_escalation(row):
        intent = row['true_intent']
        text = str(row['customer_text']).lower().strip()
        
        if intent in ["customer_service_escalation", "other_unknown"]:
            return 1
            
        if any(p in text for p in ["stolen", "missing", "lost", "human", "agent", "real person", "access my account"]):
            return 1
            
        return 0

    df['expected_escalation'] = df.apply(calculate_expected_escalation, axis=1)
    df.to_csv(gold_path, index=False)
    print(f"Added expected_escalation to {len(df)} rows in {gold_path}.")
    print(f"Escalated cases: {df['expected_escalation'].sum()}")

if __name__ == "__main__":
    add_expected_escalation()
