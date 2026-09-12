"""
scripts/label_golden_set.py
---------------------------
Interactive terminal UI to manually label the Golden Set.
Reads data/gold/amazonhelp_golden_set.csv and configs/intent_taxonomy.yaml.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import yaml

# ──────────────────────────────────────────────
CSV_PATH = Path("data/gold/amazonhelp_golden_set.csv")
YAML_PATH = Path("configs/intent_taxonomy.yaml")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_taxonomy():
    if not YAML_PATH.exists():
        print(f"[ERROR] Taxonomy not found at {YAML_PATH}")
        sys.exit(1)
    with open(YAML_PATH, "r") as f:
        data = yaml.safe_load(f)
    return [intent["name"] for intent in data.get("intents", [])]

def main():
    if not CSV_PATH.exists():
        print(f"[ERROR] Golden Set not found at {CSV_PATH}")
        sys.exit(1)

    intents = load_taxonomy()
    if not intents:
        print("[ERROR] No intents found in taxonomy.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    
    # Find unlabeled examples
    unlabeled_mask = df["intent"] == "UNLABELED"
    total = len(df)
    labeled = total - unlabeled_mask.sum()
    
    if labeled == total:
        print(f"🎉 All {total} examples have been labeled!")
        return

    print(f"Starting labeling session. Progress: {labeled}/{total}")
    print("Press Ctrl+C anytime to save and exit.\n")
    input("Press ENTER to start...")

    try:
        for idx in df[unlabeled_mask].index:
            clear_screen()
            row = df.loc[idx]
            
            print(f"Progress: {labeled}/{total} ({(labeled/total)*100:.1f}%)")
            print(f"Example ID: {row['example_id']}\n")
            
            # Print customer text safely
            safe_text = str(row['customer_text']).encode("ascii", errors="replace").decode("ascii")
            print("-" * 60)
            print(f"CUSTOMER: {safe_text}")
            print("-" * 60)
            print("\nAvailable Intents:")
            for i, intent in enumerate(intents, 1):
                print(f"  {i}. {intent}")
            
            # Get Intent
            while True:
                choice = input("\nSelect intent number (or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    print("\nSaving and exiting...")
                    df.to_csv(CSV_PATH, index=False)
                    sys.exit(0)
                if choice.isdigit() and 1 <= int(choice) <= len(intents):
                    selected_intent = intents[int(choice)-1]
                    break
                print("Invalid choice. Try again.")

            # Get Difficulty
            while True:
                diff = input("Difficulty (1=easy, 2=medium, 3=hard): ").strip()
                if diff in ['1', '2', '3']:
                    diff_map = {'1': 'easy', '2': 'medium', '3': 'hard'}
                    selected_diff = diff_map[diff]
                    break
                print("Invalid choice. Try again.")

            # Get Notes
            notes = input("Notes (optional, press Enter to skip): ").strip()

            # Save to dataframe
            df.at[idx, "intent"] = selected_intent
            df.at[idx, "difficulty"] = selected_diff
            df.at[idx, "labeling_notes"] = notes
            
            labeled += 1
            # Save progress every step so nothing is lost
            df.to_csv(CSV_PATH, index=False)
            
    except KeyboardInterrupt:
        print("\n\nSaving and exiting...")
        df.to_csv(CSV_PATH, index=False)
        sys.exit(0)

    print(f"\n🎉 All {total} examples have been labeled!")

if __name__ == "__main__":
    main()
