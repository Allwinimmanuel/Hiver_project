# Simple Baseline Rules

The Simple Baseline (Keyword Heuristics) is a deterministic model designed to classify customer intents based strictly on the presence of specific keywords within the message text. It does not use machine learning or external APIs.

## Methodology
The text is converted to lowercase. A series of if/elif statements check for the presence of keywords. The order of evaluation matters (the first matched group wins). If no keywords are matched, it defaults to the `other_unknown` intent.

## Rules List

1. **`order_status_tracking`**:
   - Keywords: "order", "track", "status", "shipped", "when"
   - *Rationale*: Typically used when a customer wants an update on their purchase.

2. **`delivery_issue`**:
   - Keywords: "delivery", "missing", "stolen", "lost", "arrive"
   - *Rationale*: Specifically targets packages that failed to arrive or were compromised.

3. **`refund_return`**:
   - Keywords: "refund", "return", "cancel", "money back"
   - *Rationale*: Targets cancellations and money-back guarantees.

4. **`account_billing`**:
   - Keywords: "account", "login", "password", "charge", "billing"
   - *Rationale*: Issues related to accessing accounts or disputing charges.

5. **`prime_membership`**:
   - Keywords: "prime", "membership", "subscribe"
   - *Rationale*: Specific to Amazon's premium subscription service.

6. **`product_defect`**:
   - Keywords: "broken", "defect", "damage", "wrong item"
   - *Rationale*: Issues regarding the quality or correctness of the received item.

7. **`customer_service_escalation`**:
   - Keywords: "human", "agent", "real person", "escalate"
   - *Rationale*: Explicit demands to bypass automation.

8. **`other_unknown`**:
   - Keywords: *None* (Fallback)
   - *Rationale*: Used if no preceding keywords are found in the customer message.
