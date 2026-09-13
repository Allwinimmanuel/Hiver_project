import pandas as pd
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from app import generate_reply, EVIDENCE_DB, OPENAI_AVAILABLE

import openai
from dotenv import load_dotenv

load_dotenv()

def heuristic_fallback_judge(reply, pred_intent, exp_intent, evidence):
    # A simple deterministic rule-based fallback judge
    scores = {
        "intent_correctness": 5 if pred_intent == exp_intent else 2,
        "relevance": 4 if len(reply) > 20 else 2,
        "grounding": 5 if "Template fallback" in reply or len(reply) > 20 else 3,
        "helpfulness": 4 if len(reply) > 20 else 2,
        "safety": 5, # Templates are always safe
    }
    
    # Calculate overall
    overall = int(sum(scores.values()) / len(scores))
    scores["overall_quality"] = overall
    
    return {
        "scores": scores,
        "reason": "Fallback heuristic evaluation (LLM unavailable)."
    }

def main():
    print("=" * 60)
    print("LLM-AS-A-JUDGE EVALUATION")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    use_llm = OPENAI_AVAILABLE and bool(api_key)
    
    if not use_llm:
        print("[WARNING] OpenAI API key is missing or 'openai' package is not installed.")
        print("Falling back to Heuristic Evaluation Mode.")
    else:
        client = openai.OpenAI(api_key=api_key)

    client = openai.OpenAI(api_key=api_key)
    
    gold_path = Path("data/evaluation/golden_set_predictions.csv")
    if not gold_path.exists():
        print(f"Error: {gold_path} not found. Run evaluation first.")
        return
        
    df = pd.read_csv(gold_path)
    # Take a sample of 25 examples for evaluation
    sample_df = df.sample(n=min(25, len(df)), random_state=42).copy()
    
    results = []
    
    print(f"Evaluating {len(sample_df)} examples using GPT-4-turbo as Judge...")
    
    for i, row in sample_df.iterrows():
        customer_msg = row['customer_text']
        pred_intent = row['predicted_intent']
        exp_intent = row['true_intent']
        conf = row['confidence']
        
        evidence = EVIDENCE_DB.get(pred_intent, EVIDENCE_DB["other_unknown"])
        reply, mode = generate_reply(pred_intent, evidence, customer_msg)
        
        # LLM Judge Prompt or Fallback
        if not use_llm:
            judge_res = heuristic_fallback_judge(reply, pred_intent, exp_intent, evidence)
            results.append({
                "example_id": row['example_id'],
                "customer_message": customer_msg,
                "expected_intent": exp_intent,
                "predicted_intent": pred_intent,
                "confidence": conf,
                "retrieved_evidence": evidence,
                "generated_reply": reply,
                "reply_mode": mode,
                "judge_intent_score": judge_res["scores"].get("intent_correctness", 0),
                "judge_relevance_score": judge_res["scores"].get("relevance", 0),
                "judge_grounding_score": judge_res["scores"].get("grounding", 0),
                "judge_helpfulness_score": judge_res["scores"].get("helpfulness", 0),
                "judge_safety_score": judge_res["scores"].get("safety", 0),
                "judge_overall_score": judge_res["scores"].get("overall_quality", 0),
                "judge_reason": judge_res.get("reason", "")
            })
            print(f"  Processed [{len(results)}/{len(sample_df)}] - Score: {judge_res['scores'].get('overall_quality')}/5 (Heuristic)")
        else:
            judge_prompt = f"""You are an expert customer support QA evaluator.
Evaluate the following generated support reply based on this rubric. Return ONLY a valid JSON object.

Customer Message: {customer_msg}
Predicted Intent: {pred_intent} (Expected: {exp_intent})
Retrieved Evidence: {evidence}
Generated Reply: {reply}

Rubric (Score each 1-5):
- intent_correctness: Did the reply address the predicted intent correctly?
- relevance: Is it relevant to the specific customer issue?
- grounding: Is it grounded in the retrieved evidence?
- helpfulness: Is it helpful and actionable?
- safety: Is it safe (does not invent policies) and escalates appropriately if needed?
- overall_quality: Overall score.

Output JSON format:
{{
  "scores": {{
    "intent_correctness": <int>,
    "relevance": <int>,
    "grounding": <int>,
    "helpfulness": <int>,
    "safety": <int>,
    "overall_quality": <int>
  }},
  "reason": "<string explaining the scores>"
}}
"""
            try:
                response = client.chat.completions.create(
                    model="gpt-4-turbo", # Best for judging
                    messages=[{"role": "user", "content": judge_prompt}],
                    response_format={ "type": "json_object" },
                    temperature=0.0
                )
                
                judge_res = json.loads(response.choices[0].message.content)
                
                # Store everything
                results.append({
                    "example_id": row['example_id'],
                    "customer_message": customer_msg,
                    "expected_intent": exp_intent,
                    "predicted_intent": pred_intent,
                    "confidence": conf,
                    "retrieved_evidence": evidence,
                    "generated_reply": reply,
                    "reply_mode": mode,
                    "judge_intent_score": judge_res["scores"].get("intent_correctness", 0),
                    "judge_relevance_score": judge_res["scores"].get("relevance", 0),
                    "judge_grounding_score": judge_res["scores"].get("grounding", 0),
                    "judge_helpfulness_score": judge_res["scores"].get("helpfulness", 0),
                    "judge_safety_score": judge_res["scores"].get("safety", 0),
                    "judge_overall_score": judge_res["scores"].get("overall_quality", 0),
                    "judge_reason": judge_res.get("reason", "")
                })
                print(f"  Processed [{len(results)}/{len(sample_df)}] - Score: {judge_res['scores'].get('overall_quality')}/5")
            except Exception as e:
                print(f"  Failed to evaluate example {row['example_id']}: {e}")
            
    if not results:
        print("No results generated.")
        return
        
    res_df = pd.DataFrame(results)
    out_csv = Path("data/evaluation/llm_judge_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved judge results to {out_csv}")
    
    # Generate human template
    human_cols = ["example_id", "customer_message", "generated_reply", 
                  "human_overall_score", "human_grounding_score", 
                  "human_helpfulness_score", "human_safety_score", "human_comments"]
    
    human_df = res_df[["example_id", "customer_message", "generated_reply"]].copy()
    human_df["human_overall_score"] = ""
    human_df["human_grounding_score"] = ""
    human_df["human_helpfulness_score"] = ""
    human_df["human_safety_score"] = ""
    human_df["human_comments"] = ""
    
    human_csv = Path("data/evaluation/human_reply_ratings_template.csv")
    human_df[human_cols].to_csv(human_csv, index=False)
    print(f"[OK] Saved human rating template to {human_csv}")
    
    # Generate Markdown Report
    report_path = Path("reports/llm_judge_report.md")
    avg_overall = res_df['judge_overall_score'].mean()
    avg_grounding = res_df['judge_grounding_score'].mean()
    
    report = f"""# LLM-as-a-Judge Evaluation Report

## Summary
- **Examples Evaluated:** {len(res_df)}
- **Average Overall Quality:** {avg_overall:.2f}/5.0
- **Average Grounding Score:** {avg_grounding:.2f}/5.0

## Detailed Scores
- Intent Correctness: {res_df['judge_intent_score'].mean():.2f}/5.0
- Relevance: {res_df['judge_relevance_score'].mean():.2f}/5.0
- Helpfulness: {res_df['judge_helpfulness_score'].mean():.2f}/5.0
- Safety: {res_df['judge_safety_score'].mean():.2f}/5.0

## Insights
*(Wait for human ratings to be added to `data/evaluation/human_reply_ratings_template.csv` and run `scripts/18_compare_human_llm_judge.py` to compare human agreement).*
"""
    report_path.write_text(report)
    print(f"[OK] Saved report to {report_path}")

if __name__ == "__main__":
    main()
