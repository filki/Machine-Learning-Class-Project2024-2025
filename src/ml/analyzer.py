import psutil
import pandas as pd
import numpy as np
import torch
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
from typing import cast # For type hints
import cupy  # For GPU array operations
import cudf  # For GPU DataFrames 
from cuml.manifold import UMAP as cuUMAP  # GPU-accelerated UMAP
from cuml.cluster import HDBSCAN as cuHDBSCAN  # GPU-accelerated HDBSCAN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SteamReviewAnalyzer:
    def __init__(self, sentence_transformer_model: str = 'all-MiniLM-L6-v2', 
                 checkpoint_dir: str = 'checkpoints',
                 include_sentiment: bool = False,
                 device: str = None):
        self.sentence_transformer_model = sentence_transformer_model
        self.checkpoint_dir = checkpoint_dir
        self.include_sentiment = include_sentiment
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.device == 'cuda':
            logger.info(f"CUDA available: Using GPU {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"PyTorch version: {torch.__version__}")
            try:
                import cupy as cp
                logger.info(f"CuPy version: {cp.__version__}")
            except ImportError:
                logger.warning("CuPy not installed - RAPIDS acceleration unavailable")
        else:
            logger.warning("CUDA not available: Using CPU only")
        
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
                truncation=True,
                device=0 if self.device == 'cuda' else -1
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
            self.sentence_transformer = self.sentence_transformer.to(self.device)
        
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
        
        # Initialize UMAP and HDBSCAN
        if self.device == 'cuda':
            try:
                import cupy as cp
                import cudf
                from cuml.manifold import UMAP as cuUMAP
                from cuml.cluster import HDBSCAN as cuHDBSCAN
                
                # Configure GPU-accelerated models for BERTopic
                umap_model = cuUMAP(
                    n_neighbors=15,
                    n_components=5,
                    metric='cosine',
                    min_dist=0.0,
                    random_state=42
                )
                
                hdbscan_model = cuHDBSCAN(
                    min_cluster_size=10,
                    min_samples=5,
                    metric='euclidean',
                    prediction_data=True,
                    gen_min_span_tree=True,
                    cluster_selection_method='eom'
                )
                
                logger.info("Using GPU-accelerated UMAP and HDBSCAN")
            except ImportError:
                logger.warning("RAPIDS not found. Using CPU implementation")
                umap_model = UMAP(
                    n_neighbors=15,
                    n_components=5,
                    min_dist=0.0,
                    metric='cosine',
                    random_state=42
                )
                hdbscan_model = hdbscan.HDBSCAN(
                    min_cluster_size=10,
                    min_samples=5,
                    metric='euclidean',
                    prediction_data=True,
                    gen_min_span_tree=True,
                    cluster_selection_method='eom'
                )
        else:
            umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=10,
                min_samples=5,
                metric='euclidean',
                prediction_data=True,
                gen_min_span_tree=True,
                cluster_selection_method='eom'
            )
        
        # Combined stop words (gaming-specific and common English)
        gaming_stop_words = [
            'game', 'games', 'play', 'played', 'playing',
            'steam', 'review', 'reviews', 'recommend',
            'hour', 'hours'
        ]

        # Topic modeling
        self.topic_model = BERTopic(
            embedding_model=self.sentence_transformer_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=True,
            top_n_words=4,
            vectorizer_model=CountVectorizer(
                stop_words=gaming_stop_words,
                min_df=5,
                ngram_range=(1, 2)
            )
        )
        topics, probs = self.topic_model.fit_transform(sentences, embeddings)
        
        results = {
            'embeddings': embeddings,
            'topics': topics,
            'topic_model': self.topic_model,
            'probabilities': probs
        }
        
        if checkpoint_id:
            save_checkpoint(self.checkpoint_dir, 'topics', results, checkpoint_id)
        
        return results
    
    def monitor_memory(self, stage: str):
        """Monitor CPU and GPU memory usage"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
            logger.info(f"GPU Memory at {stage}: {gpu_memory:.2f} MB")
        
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024**2
        logger.info(f"CPU Memory at {stage}: {cpu_memory:.2f} MB")

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

    def run_analysis(self, reviews: pd.Series, viz_dir: str = None, resume: bool = True) -> Dict[str, Any]:
        """Run the complete analysis pipeline"""
        self.monitor_memory("start")
        checkpoint_id = generate_checkpoint_id(reviews)
        viz_dir = viz_dir or 'visualizations'
        
        sentences = self.preprocess_reviews(reviews)
        self.monitor_memory("after_preprocess")
        results = self.analyze_topics(sentences, checkpoint_id, resume)
        self.monitor_memory("after_topics")
        
        final_results = {
            'sentences': sentences,
            'results': results,
            'checkpoint_id': checkpoint_id
        }
        
        if self.include_sentiment:
            topic_sentiment = self.analyze_topic_sentiment(
                sentences,
                results['topics']
            )
            self.monitor_memory("after_sentiment")
            final_results['sentiment_analysis'] = topic_sentiment.to_dict(orient='records')
            create_sentiment_visualizations(topic_sentiment, viz_dir)
        
        create_visualizations(final_results, output_dir=viz_dir)
        self.monitor_memory("final")
        
        return final_results