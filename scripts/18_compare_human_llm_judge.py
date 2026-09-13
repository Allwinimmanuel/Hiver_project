import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("=" * 60)
    print("HUMAN-LLM AGREEMENT COMPARISON")
    print("=" * 60)
    
    judge_path = Path("data/evaluation/llm_judge_results.csv")
    human_path = Path("data/evaluation/human_reply_ratings_template.csv")
    
    if not judge_path.exists() or not human_path.exists():
        print("Missing evaluation files. Run 17_llm_judge_evaluation.py first.")
        return
        
    judge_df = pd.read_csv(judge_path)
    human_df = pd.read_csv(human_path)
    
    # Check if human ratings have been filled in
    # If all 'human_overall_score' are NaN, then they haven't been filled
    if human_df['human_overall_score'].isnull().all():
        print("[PENDING] Human agreement evidence is pending manual ratings.")
        print(f"Please fill out ratings in {human_path} and run this script again.")
        
        # Write pending status to report
        report_path = Path("reports/llm_human_agreement.md")
        report_path.write_text("# Human-LLM Agreement\n\n**Status:** Human agreement evidence is pending manual ratings.")
        return
        
    print("Analyzing agreement between LLM Judge and Human Annotator...")
    
    # Drop rows where human hasn't rated yet
    merged_df = pd.merge(judge_df, human_df, on="example_id", how="inner")
    merged_df = merged_df.dropna(subset=['human_overall_score'])
    
    if len(merged_df) == 0:
        print("[PENDING] No valid human ratings found. Pending manual ratings.")
        return
        
    # Calculate metrics
    llm_scores = merged_df['judge_overall_score'].astype(float)
    human_scores = merged_df['human_overall_score'].astype(float)
    
    exact_match = (llm_scores == human_scores).mean() * 100
    within_one = (np.abs(llm_scores - human_scores) <= 1).mean() * 100
    mad = np.abs(llm_scores - human_scores).mean()
    
    try:
        correlation = np.corrcoef(llm_scores, human_scores)[0, 1]
    except Exception:
        correlation = 0.0
        
    print(f"Evaluated {len(merged_df)} rated examples.")
    print(f"Exact Agreement: {exact_match:.1f}%")
    print(f"Agreement within 1 point: {within_one:.1f}%")
    print(f"Mean Absolute Difference: {mad:.2f}")
    
    report_path = Path("reports/llm_human_agreement.md")
    report = f"""# LLM-Human Agreement Report

## Summary
- **Rated Examples:** {len(merged_df)}
- **Exact Agreement:** {exact_match:.1f}%
- **Agreement within 1 point:** {within_one:.1f}%
- **Mean Absolute Difference (MAD):** {mad:.2f}
- **Pearson Correlation:** {correlation:.2f}

## Conclusion
The LLM Judge aligns with human evaluators within 1 point {within_one:.1f}% of the time.
"""
    report_path.write_text(report)
    print(f"\n[OK] Saved agreement report to {report_path}")

if __name__ == "__main__":
    main()
