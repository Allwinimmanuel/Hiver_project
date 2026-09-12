import pandas as pd
import re
from pathlib import Path
import sys

def weak_label(text):
    text = str(text).lower()
    
    rules = {
        "order_status_tracking": [
            r"where is my order", r"tracking number", r"order status", r"track my package",
            r"hasn't shipped", r"not shipped", r"when will it ship"
        ],
        "delivery_issue": [
            r"package late", r"delivery delayed", r"hasn't arrived", r"missing package", 
            r"never arrived", r"delivered but not here", r"wrong address", r"stolen"
        ],
        "refund_return": [
            r"refund", r"return item", r"money back", r"return my order", r"return policy",
            r"credited"
        ],
        "account_billing": [
            r"charged", r"payment issue", r"billing problem", r"account locked",
            r"unrecognized charge", r"charged twice"
        ],
        "prime_membership": [
            r"prime membership", r"cancel prime", r"prime subscription", r"prime video",
            r"prime music"
        ],
        "product_defect": [
            r"broken product", r"defective item", r"damaged product", r"stopped working",
            r"missing part", r"wrong item"
        ],
        "customer_service_escalation": [
            r"speak to a representative", r"supervisor", r"customer service complaint", 
            r"escalate issue", r"talk to someone", r"real person", r"worst service"
        ]
    }
    
    matches = []
    for intent, patterns in rules.items():
        for pat in patterns:
            if re.search(pat, text):
                matches.append(intent)
                break
                
    if len(matches) == 1:
        return matches[0], 0.9, "Single Keyword Match"
    elif len(matches) > 1:
        # Conflict resolution could be complex, for weak supervision, let's pick the first one but lower confidence
        return matches[0], 0.6, "Multiple Keyword Matches (Conflict)"
    else:
        return "other_unknown", 0.8, "No Keyword Match (Fallback)"

def main():
    in_path = Path("data/processed/amazonhelp_clean.csv")
    out_path = Path("data/processed/amazonhelp_weak_labeled.csv")
    
    print("Loading cleaned dataset...")
    try:
        df = pd.read_csv(in_path, dtype={"customer_author_id": str})
    except Exception as e:
        print(f"Error loading: {e}")
        sys.exit(1)
        
    print(f"Applying weak supervision to {len(df)} rows...")
    
    intents = []
    confidences = []
    methods = []
    
    # Using vectorized apply for speed where possible, but list comprehension is fine for this size
    for text in df["customer_text"]:
        intent, conf, method = weak_label(text)
        intents.append(intent)
        confidences.append(conf)
        methods.append(method)
        
    df["intent"] = intents
    df["confidence"] = confidences
    df["labeling_method"] = methods
    
    # We are required to output certain columns
    out_df = df[["customer_text", "support_text", "intent", "labeling_method", "confidence"]]
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    
    print(f"[OK] Weak labels saved to {out_path}")
    print(df["intent"].value_counts())

if __name__ == "__main__":
    main()
