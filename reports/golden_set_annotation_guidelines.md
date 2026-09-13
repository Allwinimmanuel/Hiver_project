# Golden Set Annotation Guidelines

## Overview
The Golden Set consists of 250 manually verified customer support examples. To accurately evaluate both intent classification and our routing logic, every example contains an `expected_escalation` label. 

## Escalation Criteria
The `expected_escalation` label determines whether a ticket should be automatically handled (0) or escalated to a human agent (1). The label is independent of the model's prediction and is based purely on the original message content and expected intent.

### When to Auto-Handle (0)
A message should be auto-handled if it meets ALL of the following criteria:
1. The request falls into a clear, supported intent (e.g., `delivery_issue`, `refund_return`, `order_status_tracking`, `account_billing`, `prime_membership`, `product_defect`).
2. The language is standard and does not include explicit demands for human intervention.
3. The issue can theoretically be resolved using our provided historical evidence and self-service actions.

### When to Escalate (1)
A message must be escalated to a human agent if it meets ANY of the following criteria:

**1. Out-of-Scope or Ambiguous Requests:**
- The message is too vague to act upon.
- The request requires a service not supported by our current automated flows.
- **Intent Mapping:** Anything mapped to `other_unknown` or `customer_service_escalation`.

**2. Sensitive or High-Risk Keywords:**
- The customer explicitly asks for a human (e.g., "human", "agent", "real person").
- The issue involves severe financial risk or theft (e.g., "stolen", "missing", "lost").
- The issue involves sensitive security access (e.g., "access my account").

### Independence from Model Prediction
The ground truth label `expected_escalation` must be derived from the text and the `true_intent`, not the `predicted_intent` or the model's confidence score. If the model makes a highly confident but wrong prediction on an ambiguous text, the text should still be labeled as requiring escalation (1).
