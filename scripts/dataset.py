import os
import json
import random
import torch
from torch.utils.data import Dataset

class NeutralDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=64, split='train'):
        """
        Args:
            data_dir (str): Path to data/neutral directory
            tokenizer: HuggingFace tokenizer
            max_length (int): Max sequence length
            split (str): 'train' or 'val'
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        files = ['iadjective.txt', 'naadjective.txt', 'nouns.txt', 'verbs.txt']
        
        all_words = []
        for filename in files:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            all_words.append(word)
            else:
                print(f"Warning: {filepath} not found.")

        # Deterministic shuffle
        random.seed(42)
        random.shuffle(all_words)
        
        # 90/10 Split
        split_idx = int(len(all_words) * 0.9)
        if split == 'train':
            self.examples = all_words[:split_idx]
        else:
            self.examples = all_words[split_idx:]
            
        print(f"NeutralDataset ({split}): {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target': torch.tensor(0.0, dtype=torch.float)
        }

class FormalityRankingDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=64, split='train'):
        """
        Args:
            jsonl_path (str): Path to formality_dataset.jsonl
            tokenizer: HuggingFace tokenizer
            max_length (int): Max sequence length
            split (str): 'train' or 'val'
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []
        
        if not os.path.exists(jsonl_path):
            print(f"Error: {jsonl_path} not found.")
            return

        all_pairs = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Support both {"sentences": [...]} and [...]
                    if isinstance(data, dict):
                        sentences = data.get('sentences', [])
                    elif isinstance(data, list):
                        sentences = data
                    else:
                        continue
                        
                    # Generate pairs (si, sj) where j > i => score(sj) > score(si)
                    # So sj is more polite than si
                    for i in range(len(sentences)):
                        for j in range(i + 1, len(sentences)):
                            # (less_polite, more_polite)
                            all_pairs.append((sentences[i], sentences[j]))
                except json.JSONDecodeError:
                    continue

        # Deterministic shuffle
        random.seed(42)
        random.shuffle(all_pairs)
        
        # Reserve 10,000 for validation
        val_size = 10000
        if len(all_pairs) <= val_size:
            print("Warning: Not enough pairs for validation split. Using 10% for val.")
            val_size = int(len(all_pairs) * 0.1)
            
        if split == 'val':
            self.pairs = all_pairs[:val_size]
        else:
            self.pairs = all_pairs[val_size:]
            
        print(f"FormalityRankingDataset ({split}): {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        less_polite, more_polite = self.pairs[idx]
        
        enc_less = self.tokenizer(
            less_polite,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        enc_more = self.tokenizer(
            more_polite,
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
