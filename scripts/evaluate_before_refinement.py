import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import json
import os
import argparse
from model import PolitenessModel

class SimpleScalarDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        enc = self.tokenizer(
            item['sentence'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'target': torch.tensor(item['target_scalar'], dtype=torch.float)
        }

class SimplePairDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        enc_less = self.tokenizer(
            item['less_polite'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        enc_more = self.tokenizer(
            item['more_polite'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'less_input_ids': enc_less['input_ids'].squeeze(0),
            'less_attention_mask': enc_less['attention_mask'].squeeze(0),
            'more_input_ids': enc_more['input_ids'].squeeze(0),
            'more_attention_mask': enc_more['attention_mask'].squeeze(0)
        }

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    # We need to know model size to load correct config. 
    # Assuming 'base' if not specified or trying to infer?
    # The prompt says "Load the model from model/ (Stage-1 output)".
    # Stage-1 output was saved as `models/model_{size}_{freeze}.pth`.
    # But the user prompt says "Stage-1 already created a model saved into: model/".
    # This might be a slight inconsistency or the user moved it. 
    # I will assume the user provides the path or I look for it.
    # The prompt for `evaluate_before_refinement.py` doesn't specify args for model path, 
    # but it says "Load the model from model/".
    # I'll add an arg for model path and size.
    
    print(f"Loading tokenizer for {args.model_size}...")
    if args.model_size == 'base':
        model_name = 'cl-tohoku/bert-base-japanese'
    else:
        model_name = 'cl-tohoku/bert-large-japanese'
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model from {args.model_path}...")
    model = PolitenessModel(model_size=args.model_size)
    
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Model file {args.model_path} not found.")
        return

    model.to(device)
    model.eval()

    # Datasets
    scalar_test_path = 'data/processed/keico_scalar_test.jsonl'
    pairs_test_path = 'data/processed/keico_pairs_test.jsonl'
    
    if not os.path.exists(scalar_test_path) or not os.path.exists(pairs_test_path):
        print("Error: Test data not found. Run preprocess_keico.py first.")
        return

    scalar_ds = SimpleScalarDataset(scalar_test_path, tokenizer)
    pairs_ds = SimplePairDataset(pairs_test_path, tokenizer)
    
    scalar_loader = DataLoader(scalar_ds, batch_size=32, shuffle=False)
    pairs_loader = DataLoader(pairs_ds, batch_size=32, shuffle=False)

    # Evaluation
    mse_loss_fn = nn.MSELoss()
    total_mse = 0
    count_scalar = 0
    
    print("Evaluating Scalar MSE...")
    with torch.no_grad():
        for batch in scalar_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)
            
            scores = model(ids, mask).squeeze()
            total_mse += mse_loss_fn(scores, target).item() * ids.size(0)
            count_scalar += ids.size(0)
            
    avg_mse = total_mse / count_scalar if count_scalar > 0 else 0
    
    print("Evaluating Ranking Accuracy...")
    correct_pairs = 0
    count_pairs = 0
    
    with torch.no_grad():
        for batch in pairs_loader:
            less_ids = batch['less_input_ids'].to(device)
            less_mask = batch['less_attention_mask'].to(device)
            more_ids = batch['more_input_ids'].to(device)
            more_mask = batch['more_attention_mask'].to(device)
            
            s_less = model(less_ids, less_mask).squeeze()
            s_more = model(more_ids, more_mask).squeeze()
            
            correct = (s_more > s_less).float().sum().item()
            correct_pairs += correct
            count_pairs += less_ids.size(0)
            
    avg_acc = correct_pairs / count_pairs if count_pairs > 0 else 0
    
    print("-" * 40)
    print(f"Coarse model scalar MSE: {avg_mse:.4f}")
    print(f"Coarse model ranking accuracy: {avg_acc:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default='base', choices=['base', 'large'])
    parser.add_argument("--model_path", type=str, required=True, help="Path to Stage-1 model checkpoint")
    
    args = parser.parse_args()
    evaluate(args)
