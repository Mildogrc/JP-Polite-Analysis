import re
from typing import Dict, Tuple, List
from janome.tokenizer import Tokenizer

class FormalityLabeler:
    """
    Rule-based formality labeler for Japanese text.
    Detects Casual, Polite, Humble, and Honorific markers.
    """
    
    def __init__(self):
        self.tokenizer = Tokenizer()
        
        # Regex patterns for markers
        self.patterns = {
            "casual": [
                r"(だ|か|よ|ね|な|ぜ|ぞ)$", # Sentence ending particles (plain)
                r"^(?!.*(です|ます|ございます)).*$" # Lack of polite copula (simplified)
            ],
            "polite": [
                r"(です|ます|でしょうか|ください)",
                r"お.+ください"
            ],
            "humble": [
                r"(いたします|いただきます|参る|申す|存じる|お.+する|ご.+する)",
                r"拝見"
            ],
            "honorific": [
                r"(召し上がる|なさる|いらっしゃる|おっしゃる|になれる|される|られる)",
                r"お.+になる|ご.+になる"
            ]
        }

    def analyze_sentence(self, sentence: str) -> Dict:
        """
        Analyzes a sentence and returns formality scores and label.
        """
        scores = {
            "casual": 0,
            "polite": 0,
            "humble": 0,
            "honorific": 0
        }
        
        # 1. Regex matching
        for style, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, sentence):
                    scores[style] += 1
        
        # 2. Morphological analysis (Janome) for more precise checks
        # (Optional: refine scores based on part-of-speech)
        tokens = list(self.tokenizer.tokenize(sentence))
        for token in tokens:
            base_form = token.base_form
            pos = token.part_of_speech
            
            # Example: Check for specific verbs
            if base_form in ["行く", "来る", "食べる"] and "動詞" in pos:
                 # Neutral/Casual base
                 pass
            
            if base_form in ["参る", "申す", "致す"]:
                scores["humble"] += 1
            
            if base_form in ["いらっしゃる", "おっしゃる", "なさる"]:
                scores["honorific"] += 1
                
            if "助動詞" in pos and base_form in ["ます", "です"]:
                scores["polite"] += 1

        # Determine dominant label
        # Priority: Honorific > Humble > Polite > Casual
        # This is a simplification; real hierarchy is complex.
        
        label = "Casual"
        max_score = 0
        
        # Normalize to 0-1 range for regression target
        # Heuristic mapping: Casual=0.0, Polite=0.33, Humble=0.66, Honorific=1.0
        # Or just use the counts to decide.
        
        raw_score = 0.0
        
        if scores["honorific"] > 0:
            label = "Honorific"
            raw_score = 1.0
        elif scores["humble"] > 0:
            label = "Humble"
            raw_score = 0.7
        elif scores["polite"] > 0:
            label = "Polite"
            raw_score = 0.4
        else:
            label = "Casual"
            raw_score = 0.1
            
        # Adjust raw_score slightly based on density
        total_markers = sum(scores.values())
        if total_markers > 0:
            density = total_markers / len(tokens) if tokens else 0
            # Small boost for density
            raw_score += min(0.1, density)
            
        return {
            "formality_auto_label": label,
            "formality_auto_score": min(1.0, raw_score),
            "marker_counts": scores
        }

if __name__ == "__main__":
    labeler = FormalityLabeler()
    examples = [
        "これはペンです。",
        "飯食いに行こうぜ。",
        "先生がいらっしゃいました。",
        "私が参ります。"
    ]
    for ex in examples:
        print(f"{ex}: {labeler.analyze_sentence(ex)}")
