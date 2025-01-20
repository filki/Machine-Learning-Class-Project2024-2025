import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from umap import UMAP
import hdbscan
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import logging
from datetime import datetime
import os
from typing import List, Dict, Any
from tqdm import tqdm
from .model_config import ModelConfig
from .utils import generate_checkpoint_id, save_checkpoint, load_checkpoint, preprocess_reviews
from .visualizations import create_visualizations, create_sentiment_visualizations

class SteamReviewAnalyzer:
    def __init__(self, config: ModelConfig = ModelConfig(), 
                 checkpoint_dir: str = 'checkpoints',
                 include_sentiment: bool = False):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.include_sentiment = include_sentiment
        
        # Models (initialized when needed)
        self.sentence_transformer = None
        self.umap_model = None
        self.cluster_model = None
        self.topic_model = None
        self.sentiment_pipeline = None
        
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _init_sentiment(self):
        """Initialize sentiment analysis pipeline if needed"""
        if self.sentiment_pipeline is None and self.include_sentiment:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                max_length=512,
                truncation=True
            )

    def analyze_sentiment(self, texts: List[str], batch_size: int = 32) -> List[dict]:
        """Analyze sentiment for a batch of texts"""
        self._init_sentiment()
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch = texts[i:i + batch_size]
            batch_results = self.sentiment_pipeline(batch)
            results.extend(batch_results)
        return results

    def analyze_topic_sentiment(self, sentences: list, topics: list) -> pd.DataFrame:
        """Analyze sentiment for each topic"""
        if not self.include_sentiment:
            return None

        df = pd.DataFrame({
            'sentence': sentences,
            'topic': topics
        })
        
        sentiments = self.analyze_sentiment(sentences)
        df['sentiment'] = [s['label'] for s in sentiments]
        df['sentiment_score'] = [s['score'] for s in sentiments]
        df['sentiment_value'] = df['sentiment'].map({'POSITIVE': 1, 'NEGATIVE': -1})
        
        topic_sentiment = df.groupby('topic').agg({
            'sentiment_value': ['mean', 'count'],
            'sentiment_score': ['mean', 'std']
        }).round(3)
        
        topic_sentiment.columns = ['sentiment_mean', 'sentence_count', 'confidence_mean', 'confidence_std']
        return topic_sentiment.reset_index()

    def create_embeddings(self, sentences: list, checkpoint_id: str = None, resume: bool = True) -> np.ndarray:
        """Create embeddings for the input sentences"""
        if checkpoint_id and resume:
            data, loaded = load_checkpoint(self.checkpoint_dir, 'embeddings', checkpoint_id)
            if loaded:
                return data
        
        if self.sentence_transformer is None:
            self.sentence_transformer = SentenceTransformer(self.config.sentence_transformer_model)
        
        embeddings = []
        batch_size = 32
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="Creating embeddings"):
            batch = sentences[i:i + batch_size]
            batch_embeddings = self.sentence_transformer.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        if checkpoint_id:
            save_checkpoint(self.checkpoint_dir, 'embeddings', embeddings, checkpoint_id)
            
        return embeddings

    def analyze_topics(self, sentences: list, checkpoint_id: str = None, resume: bool = True) -> dict:
        """Perform topic analysis on the sentences"""
        if checkpoint_id and resume:
            data, loaded = load_checkpoint(self.checkpoint_dir, 'topics', checkpoint_id)
            if loaded:
                return data
        
        # Create embeddings
        embeddings = self.create_embeddings(sentences, checkpoint_id, resume)
        
        # Topic modeling with original settings
        self.topic_model = BERTopic(
            embedding_model=self.transformer_model,  # Use direct model name
            verbose=True,
            top_n_words=4,
            vectorizer_model=CountVectorizer(
                stop_words="english",
                min_df=5,   # Minimum document frequency
                ngram_range=(1, 2)  # Allow bigrams
            )
        )
        topics, probs = self.topic_model.fit_transform(sentences)
        
        results = {
            'embeddings': embeddings,
            'topics': topics,
            'topic_model': self.topic_model,
            'probabilities': probs
        }
        
        if checkpoint_id:
            save_checkpoint(self.checkpoint_dir, 'topics', results, checkpoint_id)
        
        return results

    def run_analysis(self, reviews: pd.Series, viz_dir: str = None, resume: bool = True) -> Dict[str, Any]:
        """Run the complete analysis pipeline"""
        # Generate checkpoint ID
        checkpoint_id = generate_checkpoint_id(reviews)
        
        # Use provided viz_dir or default to 'visualizations'
        viz_dir = viz_dir or 'visualizations'
        
        # Run analysis
        sentences = preprocess_reviews(reviews)
        results = self.analyze_topics(sentences, checkpoint_id, resume)
        
        final_results = {
            'sentences': sentences,
            'results': results,
            'checkpoint_id': checkpoint_id
        }
        
        # Add sentiment analysis if enabled
        if self.include_sentiment:
            topic_sentiment = self.analyze_topic_sentiment(
                sentences,
                results['topics']
            )
            final_results['sentiment_analysis'] = topic_sentiment.to_dict(orient='records')
            create_sentiment_visualizations(topic_sentiment, viz_dir)
        
        # Create visualizations
        create_visualizations(final_results, output_dir=viz_dir)
        
        return final_results