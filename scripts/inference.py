import argparse
import os
import json
import random
import torch
from transformers import AutoTokenizer
from model import PolitenessModel

def inference(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Tokenizer
    if args.model_size == 'base':
        model_name = 'cl-tohoku/bert-base-japanese'
    else:
        model_name = 'cl-tohoku/bert-large-japanese'
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Model
    print(f"Loading model from {args.model_path}...")
    model = PolitenessModel(model_size=args.model_size, freeze_encoder=args.freeze_encoder)
    
    # Load checkpoint
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Model file {args.model_path} not found.")
        return

    model.to(device)
    model.eval()

    # Load Data
    data_path = 'data/processed/stratified_data_set.jsonl'
    print(f"Loading data from {data_path}...")
    
    all_sentences = []
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        sentences = data.get('sentences', [])
                    elif isinstance(data, list):
                        sentences = data
                    else:
                        continue
                    all_sentences.extend(sentences)
                except json.JSONDecodeError:
                    continue
    else:
        print(f"Error: Data file {data_path} not found.")
        return

    if not all_sentences:
        print("No sentences found in dataset.")
        return

    print(f"Found {len(all_sentences)} total sentences.")
    
    # Sample
    num_examples = min(args.num_examples, len(all_sentences))
    sampled_sentences = random.sample(all_sentences, num_examples)
    
    print(f"\n--- Inference Results ({num_examples} random examples) ---")
    print(f"{'Score':<10} | {'Sentence'}")
    print("-" * 80)

    with torch.no_grad():
        for sentence in sampled_sentences:
            inputs = tokenizer(
                sentence,
                max_length=64,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            score = model(input_ids, attention_mask).item()
            
            print(f"{score:.4f}     | {sentence}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default='base', choices=['base', 'large'])
    parser.add_argument("--freeze_encoder", type=str, default='true', choices=['true', 'false'])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--num_examples", type=int, default=10)
    
    args = parser.parse_args()
    
    # Convert string boolean to bool
    args.freeze_encoder = args.freeze_encoder.lower() == 'true'
    
    # Default model path if not provided
    if args.model_path is None:
        freeze_str = "frozen" if args.freeze_encoder else "unfrozen"
        args.model_path = f"models/model_{args.model_size}_{freeze_str}.pth"
        
    inference(args)
