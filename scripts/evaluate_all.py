#!/usr/bin/env python
import sys
import os
import yaml
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.compute_formality import compute_formality_scores
from src.analysis.compute_sycophancy import compute_sycophancy_scores
from src.analysis.sentiment_analysis import add_sentiment
from src.analysis.statistical_analysis import analyze_correlations, run_anova
from src.analysis.visualizations import plot_formality_distributions, plot_sycophancy_vs_formality

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    output_dir = config['analysis']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Files
    llm_output = os.path.join(output_dir, "llm_responses.jsonl")
    formality_output = os.path.join(output_dir, "llm_responses_with_formality.jsonl")
    sycophancy_output = os.path.join(output_dir, "llm_responses_analyzed.jsonl")
    final_output = os.path.join(output_dir, "final_results.jsonl")
    
    # 1. Compute Formality
    if os.path.exists(llm_output):
        compute_formality_scores(llm_output, formality_output, config)
    else:
        print(f"File not found: {llm_output}. Run run_llm_queries.py first.")
        return

    # 2. Compute Sycophancy
    compute_sycophancy_scores(formality_output, sycophancy_output, config)
    
    # 3. Sentiment (Optional, slow)
    # add_sentiment(sycophancy_output, final_output)
    # For now, just copy or rename
    import shutil
    shutil.copy(sycophancy_output, final_output)
    
    # 4. Stats & Viz
    print("Generating statistics and plots...")
    df = pd.read_json(final_output, lines=True)
    
    analyze_correlations(df)
    run_anova(df)
    
    plot_formality_distributions(df, output_dir)
    plot_sycophancy_vs_formality(df, output_dir)
    
    print(f"Evaluation complete. Results in {output_dir}")

if __name__ == "__main__":
    main()
