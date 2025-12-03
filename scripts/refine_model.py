import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
import json
from model import PolitenessModel

# Reusing Dataset classes from evaluate script logic but adapted for training
# Actually, preprocess_keico.py saves .pt files which are faster.
# Let's use those if available, otherwise fallback to jsonl.
# The prompt says "Load training splits: keico_scalar_train.jsonl".
# But also "Cache tokenized versions to speed up training".
# I will implement loading from .pt for efficiency as implied by step 7.

class CachedDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.targets = data['targets']

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'target': self.targets[idx]
        }

class CachedPairDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.less_input_ids = data['less_input_ids']
        self.less_attention_mask = data['less_attention_mask']
        self.more_input_ids = data['more_input_ids']
        self.more_attention_mask = data['more_attention_mask']

    def __len__(self):
        return len(self.less_input_ids)

    def __getitem__(self, idx):
        return {
            'less_input_ids': self.less_input_ids[idx],
            'less_attention_mask': self.less_attention_mask[idx],
            'more_input_ids': self.more_input_ids[idx],
            'more_attention_mask': self.more_attention_mask[idx]
        }

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    # We need to load the Stage-1 model first.
    # Assuming standard path or passed arg.
    # The prompt says "Load Stage-1 model from: model/".
    # I'll look for a model file in `models/` (where I saved it) or `model/` (as per prompt).
    # I'll try to find the best match or use a default.
    
    freeze_str = "frozen" if args.freeze_encoder else "unfrozen"
    # Try to find a base model to start from. 
    # Ideally, we should pass the path to the Stage-1 model.
    # I'll assume the user might pass it, or I default to `models/model_{size}_frozen.pth` (common starting point).
    # But wait, if we are refining, we might want to start from the *best* Stage-1 model.
    # Let's add an arg for --initial_model_path.
    
    print(f"Loading tokenizer for {args.model_size}...")
    if args.model_size == 'base':
        model_name = 'cl-tohoku/bert-base-japanese'
    else:
        model_name = 'cl-tohoku/bert-large-japanese'
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Initializing model...")
    model = PolitenessModel(model_size=args.model_size, freeze_encoder=args.freeze_encoder)
    
    if args.initial_model_path and os.path.exists(args.initial_model_path):
        print(f"Loading Stage-1 weights from {args.initial_model_path}...")
        state_dict = torch.load(args.initial_model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False) # strict=False in case of minor differences? Should be same.
    else:
        print("Warning: No Stage-1 model path provided or found. Starting from scratch (or just BERT).")

    model.to(device)

    # Datasets
    # Try loading cached .pt files    # Datasets
    data_dir = f'data/tokenized/{args.model_size}'
    scalar_train_pt = os.path.join(data_dir, 'keico_scalar_train.pt')
    pairs_train_pt = os.path.join(data_dir, 'keico_pairs_train.pt')
    
    if os.path.exists(scalar_train_pt) and os.path.exists(pairs_train_pt):
        print("Loading cached datasets...")
        scalar_train = CachedDataset(scalar_train_pt)
        pairs_train = CachedPairDataset(pairs_train_pt)
    else:
        print("Error: Cached datasets not found. Run preprocess_keico.py first.")
        return

    # Load Test sets for final evaluation
    scalar_test_pt = os.path.join(data_dir, 'keico_scalar_test.pt')
    pairs_test_pt = os.path.join(data_dir, 'keico_pairs_test.pt')
    
    if os.path.exists(scalar_test_pt) and os.path.exists(pairs_test_pt):
        scalar_test = CachedDataset(scalar_test_pt)
        pairs_test = CachedPairDataset(pairs_test_pt)
    else:
        print("Error: Cached test datasets not found.")
        return

    # Dataloaders
    batch_size = 32
    scalar_loader = DataLoader(scalar_train, batch_size=batch_size, shuffle=True)
    pairs_loader = DataLoader(pairs_train, batch_size=batch_size, shuffle=True)
    
    scalar_test_loader = DataLoader(scalar_test, batch_size=batch_size, shuffle=False)
    pairs_test_loader = DataLoader(pairs_test, batch_size=batch_size, shuffle=False)

    # Optimizer
    lr = 1e-4 if args.freeze_encoder else 2e-5
    optimizer = AdamW(model.parameters(), lr=lr)

    # Loss Functions
    mse_loss_fn = nn.MSELoss()
    margin_loss_fn = nn.MarginRankingLoss(margin=0.1)

    # Training Loop
    print(f"Starting refinement training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # Iterate over pairs (usually larger) and cycle scalar
        scalar_iter = iter(scalar_loader)
        
        progress_bar = tqdm(pairs_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for pair_batch in progress_bar:
            try:
                scalar_batch = next(scalar_iter)
            except StopIteration:
                scalar_iter = iter(scalar_loader)
                scalar_batch = next(scalar_iter)
            
            # Move to device
            # Scalar
            s_ids = scalar_batch['input_ids'].to(device)
            s_mask = scalar_batch['attention_mask'].to(device)
            s_target = scalar_batch['target'].to(device)
            
            # Pairs
            p_less_ids = pair_batch['less_input_ids'].to(device)
            p_less_mask = pair_batch['less_attention_mask'].to(device)
            p_more_ids = pair_batch['more_input_ids'].to(device)
            p_more_mask = pair_batch['more_attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward Scalar
            s_scores = model(s_ids, s_mask).squeeze()
            loss_scalar = mse_loss_fn(s_scores, s_target)
            
            # Forward Pairs
            score_less = model(p_less_ids, p_less_mask).squeeze()
            score_more = model(p_more_ids, p_more_mask).squeeze()
            
            target_rank = torch.ones(score_less.size(0)).to(device)
            loss_rank = margin_loss_fn(score_more, score_less, target_rank)
            
            # Combined Loss
            # L = λ1 * L_scalar + λ2 * L_rank
            # λ1 = 1.0, λ2 = 0.5
            loss = 1.0 * loss_scalar + 0.5 * loss_rank
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item(), 's_loss': loss_scalar.item(), 'r_loss': loss_rank.item()})

    # Save Refined Model
    output_dir = "refined_model"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving refined model to {output_dir}...")
    model.bert.save_pretrained(output_dir) # Save BERT config and weights
    tokenizer.save_pretrained(output_dir)  # Save tokenizer
    
    # Save the full state dict including the head
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    # Also save config.json manually if needed or rely on bert.save_pretrained?
    # bert.save_pretrained saves config.json for the BERT part.
    # The head weights are in pytorch_model.bin if we save state_dict.
    # But standard HF loading might miss the head if we just use AutoModel.
    # Our PolitenessModel class handles loading.
    
    # Final Evaluation
    print("Evaluating Refined Model on Test Set...")
    model.eval()
    
    # Scalar MSE
    total_mse = 0
    count_scalar = 0
    with torch.no_grad():
        for batch in scalar_test_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)
            
            scores = model(ids, mask).squeeze()
            total_mse += mse_loss_fn(scores, target).item() * ids.size(0)
            count_scalar += ids.size(0)
    
    avg_mse = total_mse / count_scalar if count_scalar > 0 else 0
    
    # Ranking Accuracy
    correct_pairs = 0
    count_pairs = 0
    with torch.no_grad():
        for batch in pairs_test_loader:
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
    print(f"Refined model scalar MSE: {avg_mse:.4f}")
    print(f"Refined model ranking accuracy: {avg_acc:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, required=True, choices=['base', 'large'])
    parser.add_argument("--freeze_encoder", type=str, required=True, choices=['true', 'false'])
    parser.add_argument("--initial_model_path", type=str, help="Path to Stage-1 model checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    
    args = parser.parse_args()
    
    # Convert string boolean to bool
    args.freeze_encoder = args.freeze_encoder.lower() == 'true'
    
    train(args)
