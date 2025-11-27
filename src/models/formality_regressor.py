import torch
import torch.nn as nn

class FormalityRegressor(nn.Module):
    """
    MLP regression head for formality scoring.
    Input: Sentence embedding (dim=768 for BERT base)
    Output: Formality score [0, 1]
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Output in [0, 1]
        )
        
    def forward(self, x):
        return self.net(x)
