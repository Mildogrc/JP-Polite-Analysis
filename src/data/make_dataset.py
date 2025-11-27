import json
import os
import yaml
from tqdm import tqdm
from src.data.load_corpora import CorpusLoader
from src.data.rule_based_formality_labeler import FormalityLabeler

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    output_path = config['data']['dataset_path']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    loader = CorpusLoader(config)
    labeler = FormalityLabeler()
    
    print(f"Building dataset to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        count = 0
        for item in tqdm(loader.load_all()):
            sentence = item['sentence']
            analysis = labeler.analyze_sentence(sentence)
            
            record = {
                **item,
                **analysis
            }
            
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            
            if count >= 10000: # Safety limit for demo
                break
                
    print(f"Done. Saved {count} records.")

if __name__ == "__main__":
    main()
