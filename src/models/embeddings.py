import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import numpy as np

class JapaneseEmbedder:
    """
    Wrapper for Japanese BERT/LaBSE embeddings.
    """
    def __init__(self, model_name: str = "cl-tohoku/bert-base-japanese-v3", device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of texts into embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
                
        return np.vstack(all_embeddings)

if __name__ == "__main__":
    embedder = JapaneseEmbedder()
    emb = embedder.encode(["こんにちは", "おはようございます"])
    print(emb.shape)
