# Golden Set Quality Report

## Overview
The Golden Set is a separate, holdout evaluation dataset used to evaluate the intent classification, routing accuracy, and generation quality of the AmazonHelp AI Support Agent. 

- **Total Examples:** 250
- **Total Intents:** 8
- **Source:** Held-out from `amazonhelp_clean.csv` prior to model training.

## Class Distribution
The dataset contains 250 examples spanning all 8 intents.
* order_status_tracking
* delivery_issue
* refund_return
* account_billing
* prime_membership
* product_defect
* customer_service_escalation
* other_unknown

## Escalation Distribution
- **Auto-Handled Cases (Expected Escalation = 0):** 55
- **Escalated Cases (Expected Escalation = 1):** 195

*Note: The high escalation rate is heavily influenced by the high frequency of `other_unknown` requests (simple greetings, thanks) and explicit demands for human agents in the Customer Support on Twitter dataset.*

## Annotation Methodology
1. **Bootstrapping / Weak Labeling:** The `true_intent` labels were initially bootstrapped using weak supervision rules (keyword matching) to rapidly assign initial labels to a large dataset.
2. **Review & Validation:** A small subset (the Golden Set) was held out for manual review. 
3. **Escalation Labeling:** The `expected_escalation` label was generated systematically based on the presence of sensitive keywords ("stolen", "missing", "human", etc.) and the `true_intent`, strictly following the annotation guidelines.

## Potential Label Noise
While the Golden Set represents our highest-quality ground truth, it relies heavily on weak supervision bootstrapping. 
- **Limitation:** Not every single one of the 250 examples was double-blind annotated by multiple human reviewers from scratch. Some noise from the weak supervision pipeline may persist.

## Data Leakage Check
- **Result:** No Leakage Found.
- **Verification:** The 250 examples in the Golden Set were strictly removed from the training dataset prior to executing the `TfidfVectorizer.fit_transform()` step. The baseline models and the final agent have never seen these specific messages during training.
