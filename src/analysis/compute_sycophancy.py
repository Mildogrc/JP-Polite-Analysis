import json
import yaml
from src.models.sycophancy_classifier import SycophancyClassifier
from src.models.embeddings import JapaneseEmbedder
from tqdm import tqdm

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def compute_sycophancy_scores(input_file, output_file, config):
    # Optional: pass embedder for better accuracy
    embedder = JapaneseEmbedder(model_name=config['models']['embedding_model'])
    classifier = SycophancyClassifier(embedder)
    
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
            
    print("Computing sycophancy scores...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(results):
            response = item.get('response', '')
            scores = classifier.score(response)
            
            item['sycophancy_score'] = scores['sycophancy_score']
            item['sycophancy_details'] = scores
            
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    config = load_config()
    input_path = config['analysis']['output_dir'] + "/llm_responses_with_formality.jsonl"
    output_path = config['analysis']['output_dir'] + "/llm_responses_analyzed.jsonl"
    compute_sycophancy_scores(input_path, output_path, config)
