# Hiver SDE Intern Assignment: AmazonHelp AI Support Agent

A real-world AI customer support agent built on the **Customer Support on Twitter (TWCS)** dataset. This project was developed as part of the Hiver SDE Intern Assignment and adheres strictly to the 17-phase ML evaluation rubric.

---

## 1. System Architecture

The AI Support Agent is built on a hybrid architecture that prioritizes speed, accuracy, and safety over pure generative AI. 

- **The ML Classifier** handles intent prediction using a highly explainable, lightweight TF-IDF Vectorizer and Logistic Regression model trained via Weak Supervision.
- **Explicit Escalation Reasoning** handles high-priority safety and escalation cases. The system returns an explicit `escalation_reason` string (e.g., "Customer explicitly requested a human", "Sensitive account/billing issue").
- **Historical Examples** ground support replies, preventing the AI from hallucinating policies.
- **LLM Generation** is entirely optional. It leverages an OpenAI integration to draft empathetic replies, but gracefully falls back to deterministic templates if the LLM is unavailable or unconfigured.

---

## 2. Intent Classification & Baselines

- **Dataset:** 324,816 raw tweets filtered down to 168,785 clean AmazonHelp conversations.
- **Model:** TF-IDF Vectorizer + Logistic Regression with class-weight balancing.
- **Golden Evaluation Set:** 250 hand-labelled examples with `expected_escalation` labels.

**Baseline Comparisons:**
We rigorously compared our model against two baselines (see `reports/final_model_comparison.csv`):
1. **Trivial Baseline (Majority Class):** Achieves baseline macro F1 by guessing `other_unknown`.
2. **Simple Baseline (Keyword Heuristics):** Achieves moderate accuracy by strictly matching keywords without ML.
3. **Final Agent:** Outperforms both baselines significantly by combining tf-idf weighting with learned probabilities.

---

## 3. Human Escalation & Reasoning

In customer support, safety is paramount. The system implements a strict routing threshold with explicit reasoning:

- **Confidence Threshold:** Any intent predicted with `< 60% confidence` is immediately flagged for human review.
- **Keyword Overrides:** Messages containing phrases like "real person", "human", or "stolen" trigger a rule-based safety override, instantly escalating with a detailed string reason.
- **Evaluation:** Escalation logic was strictly evaluated against a ground-truth `expected_escalation` label in the Golden Set (see `reports/escalation_metrics.json`).

---

## 4. Evaluation Strategy (LLM Judge & Human)

To evaluate reply generation, we implemented an **LLM-as-a-Judge** pipeline. A robust LLM evaluates a sample of generated replies against a fixed rubric across Helpfulness, Grounding, and Safety.

To prove the validity of the LLM Judge without using synthetic ratings:
1. We generated a template for **Genuine Human Evaluation** (`data/evaluation/human_reply_ratings_template.csv`).
2. We provided a comparison script (`scripts/18_compare_human_llm_judge.py`) to calculate agreement metrics between the LLM Judge and a real human.

---

## 5. Reports & Decision Logs

Extensive documentation is provided for the research and decision-making process:
- `FINAL_RESEARCH_REPORT.md`: Comprehensive overview of the system, architecture, and results.
- `DECISION_LOG.md`: Technical decision log covering modeling, evaluation, and fallback strategies.
- `reports/failure_analysis.md`: Deep dive into real failure modes (e.g., sarcasm, OOD vocabulary), misleading headline metrics, and dataset limitations.
- `reports/golden_set_quality_report.md` & `reports/baseline_rules.md`: Documentation of the dataset and simple baseline rules.

---

## 6. Reproducibility Instructions

**1. Install Dependencies:**
```bash
py -m pip install -r requirements.txt
```

**2. Run Evaluation & Verification:**
Verify the dataset, agent, and baselines:
```bash
py scripts/14_evaluate_agent.py
py scripts/21_baseline_evaluation.py
```

**3. Run the LLM Judge & Human Comparison:**
*(Requires OpenAI API Key in `.env`)*
```bash
py scripts/17_llm_judge_evaluation.py
# After filling human_reply_ratings_template.csv manually:
py scripts/18_compare_human_llm_judge.py
```

**4. Launch the Live Agent Dashboard:**
```bash
py -m streamlit run app.py
```
*Open your browser to `http://localhost:8501` to test the AI, view the explicit escalation reasons, and inspect the API integration payload.*
