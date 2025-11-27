from transformers import pipeline
import json
from tqdm import tqdm

class SentimentAnalyzer:
    def __init__(self, model_name="koheiduck/bert-japanese-finetuned-sentiment"):
        self.pipe = pipeline("sentiment-analysis", model=model_name)

    def analyze(self, text):
        try:
            # Truncate if too long
            return self.pipe(text[:512])[0]
        except:
            return {"label": "neutral", "score": 0.5}

def add_sentiment(input_file, output_file):
    analyzer = SentimentAnalyzer()
    
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
            
    print("Computing sentiment...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in tqdm(results):
            response = item.get('response', '')
            if response:
                sent = analyzer.analyze(response)
                item['sentiment_label'] = sent['label']
                item['sentiment_score'] = sent['score']
            else:
                item['sentiment_label'] = "none"
                item['sentiment_score'] = 0.0
                
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Typically run after sycophancy
    pass
