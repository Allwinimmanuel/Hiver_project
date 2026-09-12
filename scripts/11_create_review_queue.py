import pandas as pd
from pathlib import Path
import sys

def main():
    in_path = Path("data/gold/amazonhelp_golden_set.csv")
    out_path = Path("data/gold/golden_set_review_queue.csv")

    if not in_path.exists():
        print(f"[ERROR] {in_path} not found.")
        sys.exit(1)

    df = pd.read_csv(in_path)

    # These are the exact 14 example IDs flagged as misclassified in the quality audit.
    misclassified_map = {
        "GS-120": {"suggested": "product_defect", "reason": "fraud / wrong item received"},
        "GS-036": {"suggested": "order_status_tracking", "reason": "hasn't shipped yet phrasing"},
        "GS-198": {"suggested": "delivery_issue", "reason": "second delivery to go wrong phrasing"},
        "GS-159": {"suggested": "product_defect", "reason": "Spanish producto defectuoso missed by english regex"},
        "GS-219": {"suggested": "delivery_issue", "reason": "carrier lied about services / prime context"},
        "GS-052": {"suggested": "customer_service_escalation", "reason": "poor amazon service complaint"},
        "GS-080": {"suggested": "delivery_issue", "reason": "ordered nvr delivered phrasing"},
        "GS-206": {"suggested": "delivery_issue", "reason": "package is not being delivered on time phrasing"},
        "GS-222": {"suggested": "delivery_issue", "reason": "unsafe to leave prime packages phrasing"},
        "GS-070": {"suggested": "delivery_issue", "reason": "delivery said it would arrive before 9pm phrasing"},
        "GS-082": {"suggested": "delivery_issue", "reason": "French livreur missed by english regex"},
        "GS-184": {"suggested": "order_status_tracking", "reason": "shows up under open orders phrasing"},
        "GS-156": {"suggested": "delivery_issue", "reason": "arrange delivery after 4pm phrasing"},
        "GS-164": {"suggested": "delivery_issue", "reason": "problem in pickup phrasing"}
    }

    queue_rows = []
    
    for _, row in df.iterrows():
        eid = row["example_id"]
        if eid in misclassified_map:
            queue_rows.append({
                "example_id": eid,
                "customer_text": row["customer_text"],
                "current_intent": row["intent"],
                "suggested_intent": misclassified_map[eid]["suggested"],
                "reason_for_review": misclassified_map[eid]["reason"],
                "human_decision": "",
                "final_intent": ""
            })

    out_df = pd.DataFrame(queue_rows)
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Created {out_path} with {len(out_df)} items.")

if __name__ == "__main__":
    main()
