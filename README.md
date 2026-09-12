# Hiver SDE Intern Assignment: AmazonHelp AI Support Agent

A real-world AI customer support agent built on the **Customer Support on Twitter (TWCS)** dataset. This project was developed as part of the Hiver SDE Intern Assignment.

---

## 1. Problem Framing

For a massive support handle like `@AmazonHelp`, "good" means **high-precision triage and rapid resolution**. The goal is not to have the AI have long conversational chats, but rather to quickly classify the user's intent and provide the exact link or action they need (e.g., tracking portal, returns page).

**What I chose *not* to build:**
- I did not build a generative conversational LLM chatbot. Generative models on Twitter data often hallucinate support policies. Instead, I built a deterministic classifier that maps to pre-approved, safe support actions.
- I did not use paid LLM APIs for training to prove that an end-to-end NLP pipeline can be built locally from scratch using Weak Supervision.

---

## 2. Deliverables & Pipeline

- **Dataset:** 324,816 raw tweets filtered down to 168,785 clean AmazonHelp conversations.
- **Weak Supervision Labeling:** Created ~22,000 labeled training examples using a deterministic keyword-based regex pipeline, bypassing manual labeling.
- **Golden Evaluation Set:** 250 hand-labelled examples, stratified by text length to ensure varied complexity, manually reviewed and corrected.
- **Model:** TF-IDF Vectorizer + Logistic Regression with class-weight balancing.
- **Evaluation Harness:** Metrics calculated against the human-verified Golden Set.
- **Application:** A runnable Streamlit dashboard (`app.py`) for live inference and API payload simulation.

### Running the Pipeline (Under 15 mins)

**Install Dependencies:**
```bash
py -m pip install -r requirements.txt
```

**Run the Live Agent Dashboard:**
```bash
py -m streamlit run app.py
```
*Open your browser to `http://localhost:8501` to test the AI, view the analytics, and see the API integration.*

---

## 3. Results vs. Baselines

Our trained Logistic Regression model achieves **76.0% accuracy**. How does this compare to baselines?

| Model | Accuracy | Macro F1 | Description |
|---|---|---|---|
| **Trivial Baseline (Majority Class)** | 71.6% | 10.4% | Always predicts `other_unknown`. |
| **Simple Baseline (Keyword Rules)** | 62.4% | 34.2% | The exact regex heuristics used for weak labeling. |
| **Trained AI Agent (TF-IDF + LR)** | **76.0%** | **41.1%** | Machine Learning model capturing broader n-gram context. |

*Conclusion:* The ML model successfully generalizes beyond the hardcoded rules of the simple baseline, achieving a 14% accuracy jump over the heuristics.

---

## 4. "What is misleading about my headline number?" (MANDATORY)

My headline accuracy is **76.0%**, which sounds great for a simple TF-IDF model on messy Twitter data. **However, this number is highly misleading.**

The dataset is severely imbalanced. Over 70% of the tweets in the Golden Set are `other_unknown` (general complaints, fragments, or out-of-scope banter). Because the model correctly guesses `other_unknown` most of the time, the overall accuracy is artificially inflated. 

When we look at the **Macro F1-Score (41.1%)**, the truth is revealed: the model struggles significantly on minority classes like `prime_membership` and `account_billing`. If a customer asks about a rare topic, the model defaults to `other_unknown` to be "safe." Thus, the 76% accuracy masks the model's poor recall on specific, high-value customer intents.

---

## 5. Failure Analysis: Top 5 Failure Modes

1. **Implicit Complaints (Sarcasm/Context):** 
   - *Example:* "Thanks Amazon, great job leaving my box in the rain." 
   - *Hypothesis:* TF-IDF sees "Thanks" and "great job" and predicts `other_unknown` (positive sentiment), completely missing the implicit `delivery_issue`.
2. **Short Fragments:**
   - *Example:* "DM sent."
   - *Hypothesis:* Zero semantic signal. The model defaults to `other_unknown`.
3. **Compound Intents (Multiple issues):**
   - *Example:* "My prime video isn't working and I want a refund for my late package."
   - *Hypothesis:* The model gets confused by overlapping n-grams for `prime_membership`, `refund_return`, and `delivery_issue`, often resulting in a low-confidence misclassification.
4. **Vocabulary Mismatch (Out of Vocabulary):**
   - *Example:* "The courier yeeted my parcel."
   - *Hypothesis:* Slang or rare verbs ("yeeted") were not present in the weak supervision training data, so the vectorizer ignores them.
5. **Brand Ambiguity:**
   - *Example:* "Is this compatible with Apple TV?"
   - *Hypothesis:* Customer asking a product question, but the model has no catalog knowledge, routing it to `other_unknown` instead of `product_defect`.

---

## 6. What I'd do next with one more week

1. **Implement LLM-as-a-Judge:** I would use the HuggingFace API to route a sample of predictions to a lightweight LLM (like Llama-3-8B) to grade the logic of the Logistic Regression model, comparing AI-eval against my human Golden Set labels.
2. **Embeddings instead of TF-IDF:** Swap the TF-IDF vectorizer for `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) to capture semantic meaning rather than exact word matches, solving the vocabulary mismatch failure mode.
3. **Active Learning UI:** Build a feedback loop into the Streamlit app where support agents can click "Wrong Intent" to instantly save the correction back to a retraining database.

---

## 7. Decision Log

1. **Targeted AmazonHelp:** Chosen because e-commerce has highly distinct, mutually exclusive intents (delivery, refund, product defect) compared to telecom or airlines.
2. **Capped intents at 8:** Kept the taxonomy small to ensure high precision. 50+ intents would spread the weak supervision too thin.
3. **Used Weak Supervision:** I did not have time to manually label 22,000 tweets. Writing regex heuristics to generate noisy labels was the only viable way to build a large enough training set for ML.
4. **Stratified Golden Set by Length:** Random sampling over-indexes on short "DM sent" tweets. I binned tweets by length (short, med, long) before sampling the Golden Set to ensure the model was evaluated on complex paragraphs.
5. **TF-IDF + Logistic Regression:** Chose a classical ML pipeline over fine-tuning a Transformer because it trains in seconds on a CPU, is highly explainable, and proves I understand fundamentals before reaching for an LLM API.
6. **Class Weight Balancing:** Implemented `class_weight='balanced'` in the LR model to force it to pay attention to rare intents like `prime_membership`, rather than ignoring them to optimize global accuracy.
7. **Safe Fallback Threshold (60%):** Decided that any prediction with <60% confidence must be routed to a human. In customer service, an incorrect automated action is much worse than a slight delay for a human agent.
8. **Action-Oriented Output:** Rather than drafting a conversational reply (which risks hallucination), the Streamlit app outputs a "Suggested Action" (e.g., *Generate prepaid label*). This fits better into a modern Support Agent Copilot workflow (like Hiver's shared inbox).
9. **Dark Mode UI:** Designed the Streamlit app specifically for dark mode with pastel highlights to mimic modern SaaS dashboards (reducing eye strain for support agents).
10. **JSON API Simulation:** Added an API demo tab to the UI to explicitly demonstrate how this ML model would be consumed by backend microservices, showing architectural product thinking.
