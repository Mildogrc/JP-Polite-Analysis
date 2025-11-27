import pandas as pd
import numpy as np
from scipy import stats
import json

def load_data(jsonl_path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def analyze_correlations(df):
    """
    Computes correlations between input formality and output metrics.
    """
    print("--- Correlations ---")
    # Input formality vs Output formality
    corr_formality, p_formality = stats.pearsonr(df['input_formality_score'], df['response_formality_score'])
    print(f"Input Formality vs Output Formality: r={corr_formality:.3f}, p={p_formality:.3e}")
    
    # Input formality vs Sycophancy
    corr_syc, p_syc = stats.pearsonr(df['input_formality_score'], df['sycophancy_score'])
    print(f"Input Formality vs Sycophancy: r={corr_syc:.3f}, p={p_syc:.3e}")

def run_anova(df):
    """
    ANOVA to check if output formality differs by input label group.
    """
    print("\n--- ANOVA (Output Formality by Input Label) ---")
    groups = df.groupby('input_formality_label')['response_formality_score'].apply(list)
    f_val, p_val = stats.f_oneway(*groups)
    print(f"F={f_val:.3f}, p={p_val:.3e}")

if __name__ == "__main__":
    # Example usage
    # df = load_data("results/llm_responses_analyzed.jsonl")
    # analyze_correlations(df)
    # run_anova(df)
    pass
