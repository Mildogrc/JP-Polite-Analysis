import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_formality_distributions(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='response_formality_score', hue='input_formality_label', kde=True)
    plt.title("Output Formality Distribution by Input Label")
    plt.savefig(os.path.join(output_dir, "formality_dist.png"))
    plt.close()

def plot_sycophancy_vs_formality(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='input_formality_score', y='sycophancy_score', hue='model_name')
    plt.title("Sycophancy vs Input Formality")
    plt.savefig(os.path.join(output_dir, "sycophancy_scatter.png"))
    plt.close()

if __name__ == "__main__":
    # Example usage
    pass
