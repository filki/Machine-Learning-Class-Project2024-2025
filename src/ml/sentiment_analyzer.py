from transformers import pipeline
from typing import List

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with a pre-trained model"""
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            max_length=512,
            truncation=True
        )
    
    def analyze_batch(self, texts: List[str]) -> List[dict]:
        """Analyze sentiment with progress tracking"""
        return self.sentiment_pipeline(texts, progress_bar=True)