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
from .utils import generate_checkpoint_id, save_checkpoint, load_checkpoint
from .visualizations import create_visualizations, create_sentiment_visualizations
from .bbcode_cleaner import BBCodeCleaner
import blingfire
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SteamReviewAnalyzer:
    def __init__(self, sentence_transformer_model: str = 'all-MiniLM-L6-v2', 
                 checkpoint_dir: str = 'checkpoints',
                 include_sentiment: bool = False):
        self.sentence_transformer_model = sentence_transformer_model
        self.checkpoint_dir = checkpoint_dir
        self.include_sentiment = include_sentiment
        
        # Models (initialized when needed)
        self.sentence_transformer = None
        self.topic_model = None
        self.sentiment_pipeline = None
        self.bbcode_cleaner = BBCodeCleaner(preserve_content=True)
        
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
            self.sentence_transformer = SentenceTransformer(self.sentence_transformer_model)
        
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
        
        # Combined stop words (gaming-specific and common English)
        gaming_stop_words = [
            # Gaming-specific terms
            'game', 'games', 'play', 'played', 'playing',
            'steam', 'review', 'reviews', 'recommend',
            'hour', 'hours',
            # Common English stop words
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
            'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
            'against', 'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
            'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
            'can', 'will', 'just', 'don', 'should', 'now'
        ]

        # Topic modeling with original settings
        self.topic_model = BERTopic(
            embedding_model=self.sentence_transformer_model,
            verbose=True,
            top_n_words=4,
            vectorizer_model=CountVectorizer(
                stop_words=list(gaming_stop_words),
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

    def preprocess_reviews(self, reviews: pd.Series) -> List[str]:
        """Tokenize reviews into sentences with BBCode cleaning
        
        Args:
            reviews: Series of review texts
            
        Returns:
            List[str]: List of preprocessed sentences
        """
        reviews = reviews.dropna()
        
        # Clean BBCode first
        logger.info("Cleaning BBCode tags...")
        cleaned_reviews = self.bbcode_cleaner.clean_reviews(reviews.tolist())
        
        # Log BBCode statistics
        bbcode_stats = self.bbcode_cleaner.analyze_bbcode_usage(reviews.tolist())
        logger.info("BBCode usage statistics:")
        for tag, count in bbcode_stats.items():
            if count > 0:
                logger.info(f"{tag}: {count} occurrences")
        
        # Tokenize into sentences
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

    def run_analysis(self, reviews: pd.Series, viz_dir: str = None, resume: bool = True) -> Dict[str, Any]:
        """Run the complete analysis pipeline"""
        # Generate checkpoint ID
        checkpoint_id = generate_checkpoint_id(reviews)
        
        # Use provided viz_dir or default to 'visualizations'
        viz_dir = viz_dir or 'visualizations'
        
        # Run analysis
        sentences = self.preprocess_reviews(reviews)
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