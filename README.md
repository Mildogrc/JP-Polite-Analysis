# Japanese Formality & LLM Sycophancy Analysis

## 1. Research Idea & Hypothesis

### The Core Question
**Does the politeness level of a user's prompt subconsciously influence an AI model's personality, specifically making it more sycophantic or deferential?**

In human sociolinguistics, higher formality often signals social distance or hierarchy. When a subordinate speaks to a superior in Japanese, they use *Keigo* (Honorific/Humble language). We hypothesize that Large Language Models (LLMs), having been trained on vast amounts of human text, have internalized these sociolinguistic patterns.

### Hypotheses
1.  **Formality Matching**: LLMs will mirror the user's formality level (e.g., a Casual prompt elicits a Casual response, an Honorific prompt elicits a highly Polite response).
2.  **Sycophancy Correlation**: Higher input formality will correlate with higher "sycophancy scores"—the tendency of the model to agree with the user, flatter them, or use self-deprecating language, even if not explicitly instructed to do so.

### Objectives
-   Build a pipeline to automatically label Japanese text formality (Casual, Polite, Humble, Honorific).
-   Query multiple LLMs (GPT-4, Claude, etc.) with the same content phrased in different formality levels.
-   Quantitatively measure the "sycophancy" of the responses.
-   Establish statistical evidence for the impact of prompt formality on model behavior.

---

## 2. Project Structure

The codebase is designed as a modular pipeline:

```
.
├── config.yaml             # Global configuration (paths, model names, params)
├── requirements.txt        # Python dependencies
├── scripts/                # Executable entry points
│   ├── build_dataset.py    # Step 1: Create labeled dataset
│   ├── train_formality_model.py # Step 2: Train scoring model
│   ├── run_llm_queries.py  # Step 3: Query LLMs
│   └── evaluate_all.py     # Step 4: Analyze results
└── src/
    ├── data/               # Data processing
    │   ├── load_corpora.py # Loaders for Wikipedia, Tatoeba, etc.
    │   ├── rule_based_formality_labeler.py # Heuristic labeler
    │   └── make_dataset.py # Dataset builder
    ├── models/             # Neural Models
    │   ├── embeddings.py   # BERT/LaBSE wrapper
    │   ├── formality_regressor.py # MLP for formality scoring
    │   └── sycophancy_classifier.py # Sycophancy detection logic
    ├── llm/                # LLM Interaction
    │   ├── query_models.py # Unified API (OpenAI, Anthropic)
    │   └── batch_query.py  # Batch processing with retries
    └── analysis/           # Statistics & Viz
        ├── compute_formality.py
        ├── compute_sycophancy.py
        └── statistical_analysis.py
```

---

## 3. How to Proceed

Follow these steps to reproduce the research:

### Step 0: Setup
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **API Keys**:
    Set your API keys in your environment:
    ```bash
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    ```
3.  **Configuration**:
    Edit `config.yaml` to select which models to query (e.g., `gpt-4`, `claude-2`) and where to save results.

### Step 1: Build the Dataset
Generate a dataset of Japanese sentences labeled by formality (Casual, Polite, Humble, Honorific).
```bash
python scripts/build_dataset.py
```
*Output*: `data/processed/formality_dataset.jsonl`

### Step 2: Train the Formality Regressor
Train a neural network (MLP on top of BERT embeddings) to predict a continuous formality score [0, 1]. This model will be used to evaluate the *LLM's* response formality later.
```bash
python scripts/train_formality_model.py
```
*Output*: `models/formality_regressor.pt`

### Step 3: Run LLM Experiments
Send the dataset sentences as prompts to the configured LLMs.
```bash
python scripts/run_llm_queries.py
```
*Output*: `results/llm_responses.jsonl`

### Step 4: Analyze & Visualize
Run the full evaluation pipeline. This script will:
1.  Score the LLM responses for formality (using the trained model).
2.  Score the responses for sycophancy (using heuristics and embeddings).
3.  Compute correlations (Pearson/Spearman) and run ANOVA.
4.  Generate plots in `results/`.
```bash
python scripts/evaluate_all.py
```

---

## 4. Methodology Details

### Formality Labeling (Rule-Based)
We use `janome` for morphological analysis to detect specific markers:
-   **Casual (0.1)**: Dictionary form verbs, sentence-final particles (よ, ね), lack of desu/masu.
-   **Polite (0.4)**: Standard Desu/Masu (です/ます) endings.
-   **Humble (0.7)**: Kenjougo (謙譲語) markers like いたします (itashimasu), 参る (mairu).
-   **Honorific (1.0)**: Sonkeigo (尊敬語) markers like いらっしゃる (irassharu), なさる (nasaru).

### Sycophancy Detection
We define sycophancy as excessive agreement or flattery. We measure it via:
1.  **Keyword Matching**: Detecting phrases like "You are absolutely right" (おっしゃる通りです), "Insightful" (鋭いご指摘), etc.
2.  **Semantic Similarity**: Cosine similarity between the response and a reference set of sycophantic phrases using BERT embeddings.

### Statistical Analysis
We calculate the correlation between the **Input Formality Score** (from the prompt) and the **Output Sycophancy Score**. A strong positive correlation supports the hypothesis that polite prompts induce sycophantic behavior.
