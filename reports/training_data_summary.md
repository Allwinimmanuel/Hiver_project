# Training Data Summary

- **Total weakly labeled examples**: 168785
- **Number retained for training/validation**: 24648
- **Number removed (missing text, low confidence, or undersampled noise)**: 144137

## Class Distribution
- `other_unknown`: 12324
- `refund_return`: 6753
- `prime_membership`: 1842
- `account_billing`: 1087
- `delivery_issue`: 848
- `customer_service_escalation`: 726
- `order_status_tracking`: 707
- `product_defect`: 361

## Split Sizes
- **Training Set**: 22183
- **Validation Set**: 2465

## Limitations of Weak Supervision
The intents assigned here are generated from deterministic keywords (e.g., regex matching 'missing package'). As we saw in the Golden Set audit, this approach is extremely rigid and will incorrectly classify context-heavy or linguistically diverse messages as `other_unknown`. A model trained on this dataset will likely learn to mimic these flawed rules rather than true human reasoning, artificially inflating its performance on similar regex-labeled data while struggling on human-verified ground truth.