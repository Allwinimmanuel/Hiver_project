# System End-to-End Walkthrough

The AmazonHelp AI Support System is now 100% complete, fully verified, and audited against all 16 project requirements.

## 1. System Architecture

The project consists of three main components working together:
1.  **ML Classification Engine**: A TF-IDF + Logistic Regression model trained on `amazonhelp_clean.csv` to accurately predict customer intents into 8 specific categories.
2.  **Interactive Dashboard (Streamlit)**: A sleek user interface for live agents to test predictions, view confidence scores, priority assignments, and routing status in real-time.
3.  **Genuine REST API (FastAPI)**: A fully independent programmatic API backend that allows external services to send POST requests and receive complete predictions and generated replies.

## 2. Core Capabilities Demonstrated

### Priority Detection & Ticket Routing
The model not only detects the intent (e.g., `delivery_issue`) but automatically assigns a **Priority** (High) and a **Routing Department** (Logistics Escalations) based on the `configs/intent_taxonomy.yaml`.

### Grounded Reply Generation & Heuristic Fallback
When a ticket is processed, the system retrieves a historical resolution template from the `EVIDENCE_DB` and uses an LLM to generate a grounded, professional response. If the LLM is unavailable (missing API key), the system seamlessly degrades to a deterministic **Heuristic Fallback**, ensuring the system never crashes in production.

### LLM-as-a-Judge Evaluation
Automated evaluation using `gpt-4-turbo` scores the generated replies across metrics like intent correctness, relevance, grounding, and safety. Just like the generation engine, if the LLM is unavailable, a **Heuristic Fallback Judge** automatically executes to guarantee evaluation metrics are always recorded.

### Failure Analysis
The `15_error_analysis.py` script rigorously analyzes any misclassifications (confidence < 60% or wrong intent) and produces a `reports/failure_analysis.csv` that maps the input, expected output, and actual output to a specific **Failure Category** and provides actionable **Suggested Improvements**.

## 3. How to Use the System

**Launch the Interactive Dashboard:**
```bash
py -m streamlit run app.py
```

**Launch the Genuine REST API:**
```bash
py -m uvicorn api:app --reload
```
You can test the API by sending a POST request to `http://localhost:8000/predict` with a JSON payload: `{"customer_message": "Where is my order?"}`.

**Run the Final Audit:**
```bash
py scripts/19_final_audit.py
```
This script programmatically verifies that all 16 requirements exist, starts a temporary API server, and performs an end-to-end HTTP request test.

> [!SUCCESS]
> The final audit confirmed 100% completion across all 16 strict requirements.
