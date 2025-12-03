import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

from dataset import NeutralDataset, FormalityRankingDataset
from model import PolitenessModel

def train(args):
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

    # Datasets
    print("Loading datasets...")
    neutral_train = NeutralDataset('data/neutral', tokenizer, split='train')
    neutral_val = NeutralDataset('data/neutral', tokenizer, split='val')
    
    rank_train = FormalityRankingDataset('data/processed/stratified_data_set.jsonl', tokenizer, split='train')
    rank_val = FormalityRankingDataset('data/processed/stratified_data_set.jsonl', tokenizer, split='val')

    # Dataloaders
    neutral_train_loader = DataLoader(neutral_train, batch_size=args.batch_size, shuffle=True)
    neutral_val_loader = DataLoader(neutral_val, batch_size=args.batch_size, shuffle=False)
    
    rank_train_loader = DataLoader(rank_train, batch_size=args.batch_size, shuffle=True)
    rank_val_loader = DataLoader(rank_val, batch_size=args.batch_size, shuffle=False)

    # Model
    model = PolitenessModel(model_size=args.model_size, freeze_encoder=args.freeze_encoder)
    model.to(device)

    # Optimizer
    lr = 1e-4 if args.freeze_encoder else 2e-5
    optimizer = AdamW(model.parameters(), lr=lr)

    # Loss Functions
    mse_loss_fn = nn.MSELoss()
    # MarginRankingLoss: max(0, -y * (x1 - x2) + margin)
    # We want score_more > score_less + margin
    # => score_more - score_less > margin
    # => margin - (score_more - score_less) < 0
    # PyTorch MarginRankingLoss expects inputs x1, x2 and target y.
    # Loss = max(0, -y * (x1 - x2) + margin)
    # If y=1, Loss = max(0, -(x1 - x2) + margin) = max(0, margin - (x1 - x2))
    # So we pass x1=more_polite, x2=less_polite, y=1
    margin_loss_fn = nn.MarginRankingLoss(margin=0.1)

    # Training Loop
    best_val_loss = float('inf')
    os.makedirs('models', exist_ok=True)

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        # We need to iterate over both dataloaders. 
        # Since they might have different lengths, we can zip them or cycle the shorter one.
        # Given the requirements, rank dataset is much larger (~240k) than neutral (~small).
        # Let's iterate over rank_loader and cycle neutral_loader.
        
        neutral_iter = iter(neutral_train_loader)
        
        progress_bar = tqdm(rank_train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for rank_batch in progress_bar:
            try:
                neutral_batch = next(neutral_iter)
            except StopIteration:
                neutral_iter = iter(neutral_train_loader)
                neutral_batch = next(neutral_iter)
                
            # Move to device
            # Neutral
            n_input_ids = neutral_batch['input_ids'].to(device)
            n_mask = neutral_batch['attention_mask'].to(device)
            n_target = neutral_batch['target'].to(device) # 0.0
            
            # Rank
            r_less_ids = rank_batch['less_input_ids'].to(device)
            r_less_mask = rank_batch['less_attention_mask'].to(device)
            r_more_ids = rank_batch['more_input_ids'].to(device)
            r_more_mask = rank_batch['more_attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            # Neutral
            n_scores = model(n_input_ids, n_mask).squeeze()
            loss_neutral = mse_loss_fn(n_scores, n_target)
            
            # Rank
            score_less = model(r_less_ids, r_less_mask).squeeze()
            score_more = model(r_more_ids, r_more_mask).squeeze()
            
            # Target for MarginRankingLoss is 1 because we want x1 (more) > x2 (less)
            target_rank = torch.ones(score_less.size(0)).to(device)
            loss_rank = margin_loss_fn(score_more, score_less, target_rank)
            
            # Total Loss
            loss = 1.0 * loss_neutral + 0.5 * loss_rank
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item(), 'n_loss': loss_neutral.item(), 'r_loss': loss_rank.item()})

        # Validation
        model.eval()
        val_mse = 0
        val_rank_acc = 0
        val_count_n = 0
        val_count_r = 0
        
        with torch.no_grad():
            # Neutral Val
            for batch in neutral_val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                target = batch['target'].to(device)
                
                scores = model(ids, mask).squeeze()
                val_mse += mse_loss_fn(scores, target).item() * ids.size(0)
                val_count_n += ids.size(0)
                
            # Rank Val
            for batch in rank_val_loader:
                less_ids = batch['less_input_ids'].to(device)
                less_mask = batch['less_attention_mask'].to(device)
                more_ids = batch['more_input_ids'].to(device)
                more_mask = batch['more_attention_mask'].to(device)
                
                s_less = model(less_ids, less_mask).squeeze()
                s_more = model(more_ids, more_mask).squeeze()
                
                # Correct if score_more > score_less
                correct = (s_more > s_less).float().sum().item()
                val_rank_acc += correct
                val_count_r += less_ids.size(0)

        avg_val_mse = val_mse / val_count_n if val_count_n > 0 else 0
        avg_val_acc = val_rank_acc / val_count_r if val_count_r > 0 else 0
        
        print(f"Epoch {epoch+1} Validation: Neutral MSE = {avg_val_mse:.4f}, Rank Acc = {avg_val_acc:.4f}")
        
        # Save Model
        freeze_str = "frozen" if args.freeze_encoder else "unfrozen"
        save_path = f"models/model_{args.model_size}_{freeze_str}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, required=True, choices=['base', 'large'])
    parser.add_argument("--freeze_encoder", type=str, required=True, choices=['true', 'false'])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    
    args = parser.parse_args()
    
    # Convert string boolean to bool
    args.freeze_encoder = args.freeze_encoder.lower() == 'true'
    
    train(args)
