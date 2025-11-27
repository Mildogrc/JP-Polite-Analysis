import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
from tqdm import tqdm
from src.models.embeddings import JapaneseEmbedder
from src.models.formality_regressor import FormalityRegressor
import yaml

class FormalityDataset(Dataset):
    def __init__(self, data_path, embedder):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # Pre-compute embeddings for simplicity (or do it on the fly if large)
        # For demo, we'll do on the fly or just cache a few
        self.embedder = embedder
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['sentence']
        score = item['formality_auto_score']
        # Note: In a real scenario, pre-computing embeddings is much faster
        emb = self.embedder.encode(text)[0] 
        return torch.tensor(emb, dtype=torch.float32), torch.tensor(score, dtype=torch.float32)

def train(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embedder = JapaneseEmbedder(model_name=config['models']['embedding_model'], device=device)
    dataset = FormalityDataset(config['data']['dataset_path'], embedder)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=config['models']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['models']['batch_size'])
    
    model = FormalityRegressor(input_dim=768, 
                               hidden_dim=config['models']['hidden_dim'], 
                               dropout=config['models']['dropout']).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['models']['learning_rate'])
    
    print("Starting training...")
    for epoch in range(config['models']['epochs']):
        model.train()
        train_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(X).squeeze()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X).squeeze()
                loss = criterion(output, y)
                val_loss += loss.item()
                
        print(f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Val Loss {val_loss/len(val_loader):.4f}")
        
    # Save
    os.makedirs(os.path.dirname(config['models']['regressor_save_path']), exist_ok=True)
    torch.save(model.state_dict(), config['models']['regressor_save_path'])
    print("Model saved.")

if __name__ == "__main__":
    train()
