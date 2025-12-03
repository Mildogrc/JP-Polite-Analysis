import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class PolitenessModel(nn.Module):
    def __init__(self, model_size='base', freeze_encoder=False):
        """
        Args:
            model_size (str): 'base' or 'large'
            freeze_encoder (bool): Whether to freeze BERT parameters
        """
        super(PolitenessModel, self).__init__()
        
        if model_size == 'base':
            model_name = 'cl-tohoku/bert-base-japanese'
            hidden_size = 768
        elif model_size == 'large':
            model_name = 'cl-tohoku/bert-large-japanese'
            hidden_size = 1024
        else:
            raise ValueError("model_size must be 'base' or 'large'")
            
        print(f"Loading {model_name}...")
        self.bert = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=False,
            use_safetensors=True,
        )
        
        if freeze_encoder:
            print("Freezing encoder parameters.")
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Scoring Head
        # score = tanh( W2 · GELU(W1 h + b1) + b2 )
        # We can implement this as a Sequential block
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            score: (batch_size, 1)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        score = self.head(cls_embedding)
        return score
