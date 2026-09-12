# Golden Set Quality Audit

## Objective
To investigate the heavily skewed label distribution resulting from the automated rule-based labeling script, specifically the exceptionally high count of `other_unknown` (201 out of 250 examples). 

## Methodology
A random sample of 30 examples currently labeled as `other_unknown` was extracted and manually audited to determine if they genuinely belong to this category, or if they represent misclassifications that our simple keyword rules missed.

## Findings from the 30-Example Sample

### 1. Genuine `other_unknown` (Approx. 53%)
More than half of the sampled messages legitimately belong in the `other_unknown` fallback category. The dataset contains a high amount of noise that does not fit into standard e-commerce support intents.
- **Non-English Messages**: Many tweets were in German, Spanish, Portuguese, Italian, and French (e.g., *GS-227*, *GS-234*).
- **Contests and Promotions**: Customers asking about Twitter contest winners (e.g., *GS-084*, *GS-103*, *GS-097*).
- **Feature Requests / Non-Support**: Asking to add UPI payments or asking where to send a resume (e.g., *GS-081*, *GS-215*).
- **Contextless Follow-ups / Noise**: "Yes thanks. Should I DM you my order number?" (e.g., *GS-191*, *GS-022*).

### 2. Misclassifications (Approx. 47%)
Almost half of the sampled messages were actual support requests that should have been classified into our defined taxonomy, but were missed by the rigid regex rules.
- **Missed `delivery_issue`**: Customers used phrasing that didn't precisely match the regex. For example, "delivery to go wrong" (*GS-198*) or "arrange delivery after 4 PM" (*GS-156*).
- **Missed `order_status_tracking`**: Phrasing like "hasn't shipped yet" (*GS-036*) and "shows up under open orders" (*GS-184*).
- **Missed `customer_service_escalation`**: "Poor amazon service and poor customer support" (*GS-052*).

## Conclusion
**Is our Golden Set a trustworthy benchmark for evaluating an AI customer support agent?**

**Partially, but it requires improvement before final evaluation.**
The high `other_unknown` count is heavily influenced by non-English tweets and Twitter contests, which proves that AmazonHelp data is exceptionally noisy. However, the rule-based labeling script missed nearly half of the actual support queries because human language is too variable for simple regex (which ironically proves exactly why we need an AI agent!).

Because 47% of the `other_unknown` bucket contains misclassified actionable intents, the current labels are not "Golden" enough to serve as an infallible ground truth for a final model. 

### Recommendation for Next Steps
Before training or evaluating the final agent in Milestone 4, we must either:
1. **Manually relabel** the Golden Set using the UI provided earlier.
2. **Accept the noise** as a baseline, but acknowledge that our evaluation metrics will be artificially lower since the "Ground Truth" itself is flawed by the rule-based heuristic.
