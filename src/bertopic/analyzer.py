import pandas as pd
import numpy as np
import blingfire
from transformers import pipeline
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from .bbcode_cleaner import BBCodeCleaner
from .models import get_transformer, get_bertopic
from .sentiment_analyzer import SentimentAnalyzer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SteamReviewAnalyzer:
    def __init__(self):
        self.bbcode_cleaner = BBCodeCleaner(preserve_content=True)
        self.transformer_model = None
        self.bertopic = None
        self.sentiment_analyzer = SentimentAnalyzer()

    def create_embeddings(self, sentences: list) -> np.ndarray:
        """Create embeddings for the input sentences"""
        
        if self.transformer_model is None:
            self.transformer_model = get_transformer()
        
        embeddings = []
        batch_size = 32
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="Creating embeddings"):
            batch = sentences[i:i + batch_size]
            batch_embeddings = self.transformer_model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        return embeddings

    def analyze_topics(self, sentences: list) -> dict:
        """Perform topic analysis on the sentences"""

        embeddings = self.create_embeddings(sentences)
        
        if self.bertopic is None:
            self.bertopic = get_bertopic()

        topics, probs = self.bertopic.fit_transform(sentences, embeddings)
        
        results = {
            'embeddings': embeddings,
            'topics': topics,
            'topic_model': self.bertopic,
            'probabilities': probs
        }
        return results

    def preprocess_reviews(self, reviews: pd.Series) -> List[str]:
        """Tokenize reviews into sentences with BBCode cleaning"""
        reviews = reviews.dropna()
        
        logger.info("Cleaning BBCode tags...")
        cleaned_reviews = self.bbcode_cleaner.clean_reviews(reviews.tolist())
        
        bbcode_stats = self.bbcode_cleaner.analyze_bbcode_usage(reviews.tolist())
        logger.info("BBCode usage statistics:")
        for tag, count in bbcode_stats.items():
            if count > 0:
                logger.info(f"{tag}: {count} occurrences")
        
        reviews_tokenized = []
        for review in tqdm(cleaned_reviews, desc="Preprocessing reviews"):
            try:
                sentences = blingfire.text_to_sentences(review)
                reviews_tokenized.append(sentences)
            except Exception as e:
                logger.warning(f"Error tokenizing review: {str(e)}")
                continue
        
        all_sentences = [
            sentence 
            for review in reviews_tokenized 
            for sentence in review.split('\n') 
            if sentence.strip()
        ]
        
        logger.info(f"Extracted {len(all_sentences)} sentences from {len(reviews)} reviews")
        return all_sentences
    
    def analyze_topic_sentiment(self, sentences: list, topics: list) -> pd.DataFrame:
        """Analyze sentiment for each topic"""
        
        sentiment_analyzer = SentimentAnalyzer()
        
        df = pd.DataFrame({
            'sentence': sentences,
            'topic': topics
        })

        sentiments = sentiment_analyzer.analyze_batch(sentences)
        df['sentiment'] = [s['label'] for s in sentiments]
        df['sentiment_score'] = [s['score'] for s in sentiments]
        df['sentiment_value'] = df['sentiment'].map({'POSITIVE': 1, 'NEGATIVE': -1})

        topic_sentiment = df.groupby('topic').agg({
            'sentiment_value': ['mean', 'count'],
            'sentiment_score': ['mean', 'std']
        }).round(3)

        topic_sentiment.columns = ['sentiment_mean', 'sentence_count', 'confidence_mean', 'confidence_std']
        return topic_sentiment.reset_index()

    def run_analysis(self, reviews: pd.Series) -> Dict[str, Any]:
        """Run the complete analysis pipeline"""
        sentences = self.preprocess_reviews(reviews)
        results = self.analyze_topics(sentences)
        topic_sentiment = self.analyze_topic_sentiment(sentences, results['topics'])
        
        final_results = {
            'sentences': sentences,
            'results': results,
            'sentiment_analysis': topic_sentiment.to_dict(orient='records')
        }

        return final_results