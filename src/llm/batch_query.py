import json
import os
import yaml
from tqdm import tqdm
from src.llm.query_models import LLMWrapper
from datetime import datetime

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    wrapper = LLMWrapper(config)
    
    input_path = config['data']['dataset_path']
    output_path = config['analysis']['output_dir'] + "/llm_responses.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    models_to_test = [config['llm']['openai_model']] # Add others as needed
    
    # Load dataset
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
            
    # Limit for demo
    data = data[:10] 
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data):
            original_sentence = item['sentence']
            formality_label = item['formality_auto_label']
            
            # Construct prompt variants if needed, or just use the sentence as a prompt
            # For this research: "Observe how prompt formality affects model responses"
            # We treat the sentence as a user query/statement.
            
            prompt = original_sentence
            
            for model_name in models_to_test:
                response_text = wrapper.query_with_retry(prompt, model_name)
                
                result = {
                    "input_sentence": original_sentence,
                    "input_formality_label": formality_label,
                    "input_formality_score": item['formality_auto_score'],
                    "model_name": model_name,
                    "response": response_text,
                    "timestamp": datetime.now().isoformat()
                }
                
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
