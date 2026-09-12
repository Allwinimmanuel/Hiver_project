"""
scripts/09_auto_label.py
------------------------
Automates the labeling of the Golden Set using a deterministic rule-based
classifier derived directly from the intent taxonomy.

This is a fallback mechanism since no guaranteed LLM API is available.
"""

import sys
import re
from pathlib import Path
import pandas as pd
import yaml

CSV_PATH = Path("data/gold/amazonhelp_golden_set.csv")
YAML_PATH = Path("configs/intent_taxonomy.yaml")

def load_taxonomy():
    if not YAML_PATH.exists():
        print(f"[ERROR] Taxonomy not found at {YAML_PATH}")
        sys.exit(1)
    with open(YAML_PATH, "r") as f:
        data = yaml.safe_load(f)
    return data.get("intents", [])

# Rule-based keyword matching based exactly on our taxonomy guidelines
RULES = {
    "order_status_tracking": [r"where is my order", r"has my item shipped", r"tracking", r"status of", r"when.*ship", r"shipment", r"not shipped", r"delayed"],
    "delivery_issue": [r"delivered.*nothing", r"delivered.*not", r"missing package", r"late delivery", r"wrong address", r"never arrived", r"didn't arrive", r"did not arrive", r"stolen", r"false status", r"courier", r"driver"],
    "refund_return": [r"refund", r"return", r"money back", r"how long.*refund", r"print.*label", r"return policy", r"credited"],
    "account_billing": [r"charged twice", r"unrecognized charge", r"locked", r"password", r"log into", r"login", r"payment method", r"gift card", r"balance", r"charged me", r"billed"],
    "prime_membership": [r"prime video", r"cancel prime", r"prime music", r"prime subscription", r"smart tv", r"streaming", r"app.*crash"],
    "product_defect": [r"broken", r"defective", r"wrong item", r"missing part", r"shattered", r"damaged", r"not working", r"doesn't work", r"does not work"],
    "customer_service_escalation": [r"phone number", r"call me", r"real person", r"speak to a manager", r"terrible service", r"worst customer service", r"talk to someone", r"human", r"bot"],
}

def classify_message(text):
    text_lower = str(text).lower()
    
    matches = []
    
    for intent, patterns in RULES.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                matches.append(intent)
                break # count intent once
                
    # Evaluate matches based on guidelines
    if len(matches) == 0:
        return "other_unknown", "hard", "No specific keyword rules matched. Falling back to other."
        
    if len(matches) == 1:
        return matches[0], "easy", f"Matched single rule for {matches[0]}."
        
    # Multiple matches - Guideline: Prioritize root causes or financials
    if "delivery_issue" in matches and "refund_return" in matches:
        return "delivery_issue", "medium", "Matched delivery and refund. Guideline: prioritize root cause (delivery)."
        
    if "refund_return" in matches and "product_defect" in matches:
        return "product_defect", "medium", "Matched defect and refund. Guideline: root cause is defect."
        
    if "order_status_tracking" in matches and "delivery_issue" in matches:
        return "delivery_issue", "medium", "Matched tracking and delivery issue. Assume delivery problem takes precedence."

    if "customer_service_escalation" in matches:
        # If they match something else actionable, ignore escalation
        actionable = [m for m in matches if m != "customer_service_escalation"]
        if actionable:
            return actionable[0], "medium", f"Matched {actionable[0]} and escalation. Prioritizing actionable issue."
            
    # Default fallback for multiple matches
    return matches[0], "hard", f"Multiple overlapping matches ({', '.join(matches)}). Chose first match."

def main():
    if not CSV_PATH.exists():
        print(f"[ERROR] Golden Set not found: {CSV_PATH}")
        sys.exit(1)

    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    
    total = len(df)
    
    for idx in df.index:
        sys.stdout.write(f"\rProcessing {idx+1}/{total}... ")
        sys.stdout.flush()
        
        text = df.at[idx, "customer_text"]
        intent, diff, notes = classify_message(text)
        
        df.at[idx, "intent"] = intent
        df.at[idx, "difficulty"] = diff
        df.at[idx, "labeling_notes"] = notes

    print("\nSaving results...")
    try:
        df.to_csv(CSV_PATH, index=False)
        print(f"[OK] Successfully labeled and saved {total} rows.")
    except PermissionError:
        print("\n[ERROR] Permission denied to write to the CSV file.")
        print("Please ensure you have closed 'py scripts/label_golden_set.py' in your terminal!")
        sys.exit(1)

if __name__ == "__main__":
    main()
