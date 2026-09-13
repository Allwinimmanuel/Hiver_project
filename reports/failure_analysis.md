# Failure Modes & Metrics Analysis

## 1. Five Real Failure Modes
Despite our rigorous ML evaluation, the agent still fails in certain scenarios:

**1. Misinterpreting Politeness as an Intent:**
- *Example:* "Thank you so much!"
- *Failure:* The intent classifier often lacks context and might force this into a low-confidence bucket or incorrectly route to an actual support intent. We handled this with heuristics, but the underlying ML model still struggles with pure conversational noise.

**2. Sarcasm and Irony:**
- *Example:* "Oh great, another delayed package. You guys are the best."
- *Failure:* The TF-IDF + Logistic Regression model is heavily reliant on term frequency. "Great" and "best" trick it into missing the `delivery_issue` intent, often predicting `other_unknown`.

**3. Complex, Multi-Intent Queries:**
- *Example:* "My package never arrived, and I want a refund to my original card."
- *Failure:* The text contains keywords for `delivery_issue` and `refund_return` and `account_billing`. Our classifier outputs a single intent based on whichever term has the highest tf-idf weight, losing critical context.

**4. Ambiguous "Help" Requests:**
- *Example:* "I need help with my account."
- *Failure:* The model confidently predicts `account_billing`, but without knowing the specific issue (login failure vs unauthorized charge), the generated reply might be dangerously generic.

**5. OOD (Out of Distribution) Vocabulary:**
- *Example:* Slang or abbreviations used on Twitter (e.g., "wtv", "smh", "@amazonhelp u guys r trippin").
- *Failure:* TF-IDF fails entirely on tokens not present in the training vocabulary, resulting in zero vectors and random/low-confidence classifications.

---

## 2. The Danger of "Misleading Headline Metrics"
In our `agent_evaluation.json`, we see a high overall accuracy (e.g., ~80%+ depending on the exact test set). However, this headline metric is dangerously misleading for business operations:

1. **Class Imbalance Masking:** The high accuracy is heavily driven by predicting the majority classes (like `order_status_tracking` and `delivery_issue`).
2. **Critical Failures Hidden:** A model that achieves 90% accuracy but fails 100% of the time on `customer_service_escalation` (a rare but highly sensitive class) is a catastrophic failure in production.
3. **Escalation Asymmetry:** A False Positive on escalation just wastes human agent time. A False Negative on escalation (auto-handling a high-risk angry customer) destroys brand trust. Overall accuracy treats both errors equally.

---

## 3. Dataset Limitations
The "Customer Support on Twitter" dataset is flawed for building a comprehensive support agent:
- **Length Constraint:** Tweets are extremely short. Real support tickets are often paragraphs long with screenshots.
- **Context Loss:** The dataset often only contains the customer's initial blast, lacking the nested thread or historical context of their account.
- **Brand Specificity:** Many tweets are generic complaints ("Amazon sucks") rather than actionable support requests, meaning our model learned to classify complaints rather than resolve nuanced tickets.
