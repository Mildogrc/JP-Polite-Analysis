import json
import torch
import yaml
from src.models.embeddings import JapaneseEmbedder
from src.models.formality_regressor import FormalityRegressor
from tqdm import tqdm

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def compute_formality_scores(input_file, output_file, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    embedder = JapaneseEmbedder(model_name=config['models']['embedding_model'], device=device)
    model = FormalityRegressor(input_dim=768, 
                               hidden_dim=config['models']['hidden_dim'], 
                               dropout=config['models']['dropout']).to(device)
    
    # Load weights if available
    try:
        model.load_state_dict(torch.load(config['models']['regressor_save_path'], map_location=device))
    except:
        print("Warning: Could not load trained weights. Using random weights for demo.")
    
    model.eval()
    
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
            
    print("Computing formality scores...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(results):
            response = item.get('response', '')
            if not response:
                item['response_formality_score'] = 0.0
            else:
                emb = embedder.encode(response)
                emb_tensor = torch.tensor(emb, dtype=torch.float32).to(device)
                with torch.no_grad():
                    score = model(emb_tensor).item()
                item['response_formality_score'] = score
                
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    config = load_config()
    # Assuming input comes from batch_query output
    input_path = config['analysis']['output_dir'] + "/llm_responses.jsonl"
    output_path = config['analysis']['output_dir'] + "/llm_responses_with_formality.jsonl"
    compute_formality_scores(input_path, output_path, config)
