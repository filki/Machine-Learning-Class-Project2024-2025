from transformers import pipeline
from typing import List
from tqdm import tqdm

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with a pre-trained model"""
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            max_length=512,
            truncation=True
        )
    
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[dict]:
        """Analyze sentiment for a batch of texts"""
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch = texts[i:i + batch_size]
            batch_results = self.sentiment_pipeline(batch)
            results.extend(batch_results)
        return results