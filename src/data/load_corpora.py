import os
import json
from typing import List, Dict, Generator
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

class CorpusLoader:
    """
    Loads and normalizes various Japanese corpora.
    Returns a generator of dictionaries with 'sentence' and 'source' keys.
    """
    
    def __init__(self, config: Dict):
        self.config = config

    def load_all(self) -> Generator[Dict[str, str], None, None]:
        """Yields sentences from all configured corpora."""
        yield from self.load_tatoeba()
        yield from self.load_wikipedia()
        # Add other loaders here as needed
        # yield from self.load_aozora() 
        # yield from self.load_ted()

    def load_tatoeba(self) -> Generator[Dict[str, str], None, None]:
        """Loads Japanese sentences from Tatoeba dataset (via HuggingFace)."""
        print("Loading Tatoeba...")
        try:
            # Tatoeba dataset on HF might be 'tatoeba', specific lang pair needed usually
            # Using a generic placeholder logic or specific subset if known.
            # For demonstration, we'll try to load a small subset or mock if fails.
            ds = load_dataset("tatoeba", lang1="en", lang2="ja", split="train", streaming=True)
            for item in ds:
                if 'translation' in item and 'ja' in item['translation']:
                    yield {
                        "sentence": item['translation']['ja'],
                        "source": "tatoeba"
                    }
        except Exception as e:
            print(f"Error loading Tatoeba: {e}")

    def load_wikipedia(self) -> Generator[Dict[str, str], None, None]:
        """Loads a subset of Japanese Wikipedia."""
        print("Loading Wikipedia (streaming)...")
        try:
            # streaming=True to avoid downloading full dump
            ds = load_dataset("wikipedia", "20220301.ja", split="train", streaming=True)
            count = 0
            max_sentences = 1000 # Limit for demo purposes
            
            for item in ds:
                text = item.get('text', '')
                sentences = text.split('。')
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) > 10: # Filter short segments
                        yield {
                            "sentence": sent + "。",
                            "source": "wikipedia"
                        }
                        count += 1
                if count >= max_sentences:
                    break
        except Exception as e:
            print(f"Error loading Wikipedia: {e}")

    # Placeholder for other corpora requiring local files
    def load_local_text_file(self, path: str, source_name: str) -> Generator[Dict[str, str], None, None]:
        if not os.path.exists(path):
            print(f"Path not found: {path}")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield {
                        "sentence": line,
                        "source": source_name
                    }

if __name__ == "__main__":
    # Test run
    loader = CorpusLoader({})
    count = 0
    for item in loader.load_all():
        print(item)
        count += 1
        if count > 5:
            break
