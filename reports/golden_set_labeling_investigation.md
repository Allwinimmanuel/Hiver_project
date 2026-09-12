# Golden Set Labeling Discrepancy Investigation

## 1. Root Cause of the Discrepancy
The issue was caused by a **race condition (file overwrite)** between two scripts.

1. You started the manual labeling script (`py scripts/label_golden_set.py`) in your terminal. This script loaded the CSV (which had 250 `UNLABELED` rows) into memory.
2. While that terminal was open and waiting for your input, we executed the automated script (`py scripts/09_auto_label.py`). The auto script correctly labeled all 250 rows and saved the file to disk successfully.
3. Later, you pressed `q` in your manual labeling terminal to quit. When `label_golden_set.py` exited, it executed `df.to_csv()`, taking its old in-memory dataframe (which had your 4 manual labels and 246 `UNLABELED` rows) and completely overwriting the file on disk.

This destroyed the 250 automated labels and replaced them with your manual session's state.

## 2. Exact Number of Labels Currently Saved
The `data/gold/amazonhelp_golden_set.csv` currently contains exactly **4 valid labels**:
- `other_unknown`: 2
- `order_status_tracking`: 1
- `delivery_issue`: 1

The remaining **246 rows** are set to `UNLABELED`.

## 3. Were Labels Written to Incorrect Columns?
No. All scripts (auto, manual, and validation) are referencing the exact same column (`intent`). The data structures and columns are perfectly intact.

## 4. Can the Original 250 Auto-Labels be Recovered?
Because the file was forcefully overwritten on disk, the previous auto-labels cannot be magically undeleted. However, because our automated script (`09_auto_label.py`) is completely deterministic (rule-based keywords), **we can perfectly recreate the exact same 250 labels by simply running the script again**. 

## 5. Recommended Fix
To fix this and restore all 250 labels:
1. Ensure `label_golden_set.py` is closed in all your terminals.
2. Run `py scripts/09_auto_label.py` one more time.
3. Run `py scripts/08_validate_golden_set.py` to confirm.
