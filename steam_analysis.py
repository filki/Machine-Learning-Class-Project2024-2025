import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from umap import UMAP
import hdbscan
from bertopic import BERTopic
import blingfire
from typing import List, Tuple, Any
from dataclasses import dataclass
import logging
import os
import pickle
import json
from datetime import datetime
import hashlib
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the analysis pipeline"""
    sentence_transformer_model: str = 'all-MiniLM-L6-v2'
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.1
    hdbscan_min_cluster_size: int = 15
    bertopic_model: str = "all-MiniLM-L6-v2"

class SteamReviewAnalyzer:
    def __init__(self, config: ModelConfig = ModelConfig(), checkpoint_dir: str = 'checkpoints'):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.sentence_transformer = None
        self.umap_model = None
        self.cluster_model = None
        self.topic_model = None
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _generate_checkpoint_id(self, data: pd.Series) -> str:
        """Generate a unique identifier for the dataset"""
        sample = str(data.head(3).values) + str(data.shape)
        return hashlib.md5(sample.encode()).hexdigest()[:10]

    def _save_checkpoint(self, stage: str, data: Any, checkpoint_id: str):
        """Save checkpoint data to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.checkpoint_dir}/{checkpoint_id}_{stage}_{timestamp}.pkl"
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'stage': stage,
            'filename': filename,
            'config': self.config.__dict__
        }
        
        metadata_file = f"{self.checkpoint_dir}/{checkpoint_id}_metadata.json"
        with open(metadata_file, 'a') as f:
            json.dump(metadata, f)
            f.write('\n')
            
        logger.info(f"Saved checkpoint: {filename}")

    def _load_checkpoint(self, stage: str, checkpoint_id: str) -> Tuple[Any, bool]:
        """Load the latest checkpoint for a given stage"""
        try:
            metadata_file = f"{self.checkpoint_dir}/{checkpoint_id}_metadata.json"
            if not os.path.exists(metadata_file):
                return None, False
            
            # Read all checkpoints for this stage
            checkpoints = []
            with open(metadata_file, 'r') as f:
                for line in f:
                    checkpoint = json.loads(line)
                    if checkpoint['stage'] == stage:
                        checkpoints.append(checkpoint)
            
            if not checkpoints:
                return None, False
            
            # Get the latest checkpoint
            latest = max(checkpoints, key=lambda x: x['timestamp'])
            
            # Load the data
            with open(latest['filename'], 'rb') as f:
                data = pickle.load(f)
                
            logger.info(f"Loaded checkpoint: {latest['filename']}")
            return data, True
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None, False

    def preprocess_reviews(self, reviews: pd.Series, checkpoint_id: str = None, resume: bool = True) -> List[str]:
        """Tokenize reviews into sentences with checkpointing"""
        if checkpoint_id and resume:
            data, loaded = self._load_checkpoint('preprocess', checkpoint_id)
            if loaded:
                return data
        
        logger.info("Preprocessing reviews...")
        reviews = reviews.dropna()
        reviews_tokenized = []
        
        for i, review in enumerate(tqdm(reviews, desc="Preprocessing")):
            sentences = blingfire.text_to_sentences(review)
            reviews_tokenized.append(sentences)
            
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(reviews)} reviews")
        
        all_sentences = [
            sentence 
            for review in reviews_tokenized 
            for sentence in review.split('\n') 
            if sentence.strip()
        ]
        
        if checkpoint_id:
            self._save_checkpoint('preprocess', all_sentences, checkpoint_id)
        
        logger.info(f"Extracted {len(all_sentences)} sentences from {len(reviews)} reviews")
        return all_sentences

    def create_embeddings(self, sentences: List[str], checkpoint_id: str = None, resume: bool = True) -> np.ndarray:
        """Create embeddings with checkpointing"""
        if checkpoint_id and resume:
            data, loaded = self._load_checkpoint('embeddings', checkpoint_id)
            if loaded:
                return data
        
        logger.info(f"Creating embeddings using {self.config.sentence_transformer_model}")
        if self.sentence_transformer is None:
            self.sentence_transformer = SentenceTransformer(self.config.sentence_transformer_model)
        
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="Creating embeddings"):
            batch = sentences[i:i + batch_size]
            batch_embeddings = self.sentence_transformer.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
            
            if i % (batch_size * 100) == 0:
                logger.info(f"Processed {i}/{len(sentences)} sentences")
        
        embeddings = np.vstack(embeddings)
        
        if checkpoint_id:
            self._save_checkpoint('embeddings', embeddings, checkpoint_id)
            
        return embeddings

    def analyze_topics(self, sentences: List[str], checkpoint_id: str = None, resume: bool = True) -> dict:
        """Topic analysis with checkpointing"""
        if checkpoint_id and resume:
            data, loaded = self._load_checkpoint('topics', checkpoint_id)
            if loaded:
                return data
        
        # Create embeddings (with its own checkpointing)
        embeddings = self.create_embeddings(sentences, checkpoint_id, resume)
        
        # UMAP reduction
        logger.info("Performing UMAP dimension reduction...")
        umap_model = UMAP(
            n_neighbors=self.config.umap_n_neighbors,
            n_components=self.config.umap_n_components,
            min_dist=self.config.umap_min_dist,
            metric='cosine'
        )
        umap_embeddings = umap_model.fit_transform(embeddings)
        
        # Clustering
        logger.info("Performing clustering...")
        cluster_model = hdbscan.HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        clusters = cluster_model.fit_predict(umap_embeddings)
        
        # Topic modeling
        logger.info("Performing topic modeling...")
        topic_model = BERTopic(
            embedding_model=self.config.bertopic_model,
            verbose=True,
            # Automatic topic naming
            top_n_words=4,  # Number of words to use for naming
            vectorizer_model=CountVectorizer(stop_words="english") 
        )
        topics, probs = topic_model.fit_transform(sentences)
        
        results = {
            'embeddings': embeddings,
            'clusters': clusters,
            'topics': topics,
            'topic_model': topic_model,
            'probabilities': probs
        }
        
        if checkpoint_id:
            self._save_checkpoint('topics', results, checkpoint_id)
        
        return results

    def create_visualizations(self, results: dict, output_dir: str = 'visualizations'):
        """Create and save visualizations of the analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Topic Visualization
        logger.info("Creating topic visualization...")
        topic_model = results['results']['topic_model']
        
        # Get topic info
        topic_info = topic_model.get_topic_info()
        top_topics = topic_info.head(10)  # Get top 10 topics
        
        # Plot topic sizes
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_topics)), top_topics['Count'])
        plt.title('Top 10 Topics by Size')
        plt.xlabel('Topic ID')
        plt.ylabel('Number of Documents')
        plt.xticks(range(len(top_topics)), top_topics['Topic'])
        plt.tight_layout()
        plt.savefig(f"{output_dir}/topic_sizes_{timestamp}.png")
        plt.close()
        
        # 2. Topic Word Cloud
        try:
            for topic_id in top_topics['Topic'][:5]:  # Top 5 topics
                words = dict(topic_model.get_topic(topic_id))
                if words:
                    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS, min_word_length=3).generate_from_frequencies(words)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title(f'Topic {topic_id} Word Cloud')
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/topic_{topic_id}_wordcloud_{timestamp}.png")
                    plt.close()
        except ImportError:
            logger.warning("WordCloud package not installed. Skipping word clouds.")
        
        # 3. Cluster Visualization
        logger.info("Creating cluster visualization...")
        umap_vis = UMAP(n_components=2, random_state=42)
        vis_embeddings = umap_vis.fit_transform(results['results']['embeddings'])
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            vis_embeddings[:, 0], 
            vis_embeddings[:, 1], 
            c=results['results']['clusters'],
            cmap='tab20',
            alpha=0.6
        )
        plt.colorbar(scatter, label='Cluster')
        plt.title('Document Clusters Visualization')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/clusters_{timestamp}.png")
        plt.close()
        
        # 4. Topic-Cluster Relationship
        logger.info("Creating topic-cluster relationship visualization...")
        topic_cluster_df = pd.DataFrame({
            'Topic': results['results']['topics'],
            'Cluster': results['results']['clusters']
        })
        
        topic_cluster_counts = pd.crosstab(
            topic_cluster_df['Topic'],
            topic_cluster_df['Cluster']
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(topic_cluster_counts, cmap='YlOrRd', annot=True, fmt='d')
        plt.title('Topic-Cluster Relationship')
        plt.xlabel('Cluster ID')
        plt.ylabel('Topic ID')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/topic_cluster_relationship_{timestamp}.png")
        plt.close()
        
        return f"{output_dir}/results_{timestamp}"

    def save_detailed_results(self, results: dict, output_dir: str = 'results'):
        """Save detailed analysis results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full results pickle
        pickle_path = f"{output_dir}/full_results_{timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Create summary DataFrame
        topic_model = results['results']['topic_model']
        topic_info = topic_model.get_topic_info()
        
        # Save topic information
        topic_info.to_csv(f"{output_dir}/topic_info_{timestamp}.csv", index=False)
        
        # Save topic details
        topic_details = []
        for topic_id in topic_info['Topic']:
            if topic_id != -1:  # Skip outlier topic
                words = topic_model.get_topic(topic_id)
                topic_details.append({
                    'topic_id': topic_id,
                    'words': [word for word, _ in words[:10]],  # Top 10 words
                    'word_scores': [score for _, score in words[:10]]
                })
        
        topic_details_df = pd.DataFrame(topic_details)
        topic_details_df.to_csv(f"{output_dir}/topic_details_{timestamp}.csv", index=False)
        
        # Save cluster information
        cluster_info = pd.DataFrame({
            'cluster': results['results']['clusters'],
            'topic': results['results']['topics']
        })
        cluster_stats = cluster_info.groupby('cluster').agg({
            'topic': ['count', 'nunique']
        })
        cluster_stats.columns = ['size', 'unique_topics']
        cluster_stats.to_csv(f"{output_dir}/cluster_stats_{timestamp}.csv")
        
        # Save summary text
        with open(f"{output_dir}/analysis_summary_{timestamp}.txt", 'w') as f:
            f.write(f"Analysis Summary\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(f"Total documents analyzed: {len(results['sentences'])}\n")
            f.write(f"Number of topics found: {len(topic_info)}\n")
            f.write(f"Number of clusters found: {len(set(results['results']['clusters']))}\n")
            f.write(f"\nTop 5 Topics:\n")
            for _, row in topic_info.head().iterrows():
                f.write(f"Topic {row['Topic']}: {row['Name']} (Size: {row['Count']})\n")
        
        logger.info(f"Saved detailed results to {output_dir}/")
        return f"{output_dir}/analysis_summary_{timestamp}.txt"

    def run_analysis(self, reviews: pd.Series, resume: bool = True) -> dict:
        """Run complete analysis pipeline with checkpointing and save results"""
        # Generate checkpoint ID for this dataset
        checkpoint_id = self._generate_checkpoint_id(reviews)
        logger.info(f"Starting analysis with checkpoint ID: {checkpoint_id}")
        
        # Derive output directories from checkpoint directory
        base_dir = self.checkpoint_dir.split('_checkpoints')[0]
        viz_dir = f"{base_dir}_visualizations" if '_checkpoints' in self.checkpoint_dir else 'visualizations'
        results_dir = f"{base_dir}_results" if '_checkpoints' in self.checkpoint_dir else 'results'
        
        # Run pipeline with checkpointing
        sentences = self.preprocess_reviews(reviews, checkpoint_id, resume)
        results = self.analyze_topics(sentences, checkpoint_id, resume)
        
        final_results = {
            'sentences': sentences,
            'results': results,
            'checkpoint_id': checkpoint_id
        }
        
        # Create visualizations in the appropriate directory
        viz_path = self.create_visualizations(final_results, output_dir=viz_dir)
        logger.info(f"Created visualizations in {viz_path}")
        
        # Save detailed results in the appropriate directory
        summary_path = self.save_detailed_results(final_results, output_dir=results_dir)
        logger.info(f"Saved detailed results. Summary available at: {summary_path}")
        
        return final_results
    

from transformers import pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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

class EnhancedSteamReviewAnalyzer(SteamReviewAnalyzer):
    def __init__(self, config: ModelConfig = ModelConfig(), checkpoint_dir: str = 'checkpoints'):
        super().__init__(config, checkpoint_dir)
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def _analyze_topic_sentiment(self, sentences: List[str], topics: List[int]) -> pd.DataFrame:
        """Analyze sentiment for each topic"""
        # Create DataFrame with sentences and their topics
        df = pd.DataFrame({
            'sentence': sentences,
            'topic': topics
        })
        
        # Get sentiment for all sentences
        sentiments = self.sentiment_analyzer.analyze_batch(sentences)
        df['sentiment'] = [s['label'] for s in sentiments]
        df['sentiment_score'] = [s['score'] for s in sentiments]
        
        # Convert sentiment labels to numeric scores
        df['sentiment_value'] = df['sentiment'].map({'POSITIVE': 1, 'NEGATIVE': -1})
        
        # Calculate topic sentiment metrics
        topic_sentiment = df.groupby('topic').agg({
            'sentiment_value': ['mean', 'count'],
            'sentiment_score': ['mean', 'std']
        }).round(3)
        
        topic_sentiment.columns = ['sentiment_mean', 'sentence_count', 'confidence_mean', 'confidence_std']
        return topic_sentiment.reset_index()
    
    def create_sentiment_visualizations(self, topic_sentiment: pd.DataFrame, output_dir: str):
        """Create visualizations for sentiment analysis results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Topic Sentiment Distribution
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=topic_sentiment,
            x='topic',
            y='sentiment_mean',
            color='skyblue'
        )
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.title('Average Sentiment by Topic')
        plt.xlabel('Topic ID')
        plt.ylabel('Average Sentiment (-1 to 1)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/topic_sentiment_{timestamp}.png")
        plt.close()
        
        # 2. Topic Sentiment Confidence
        plt.figure(figsize=(12, 6))
        plt.errorbar(
            topic_sentiment['topic'],
            topic_sentiment['confidence_mean'],
            yerr=topic_sentiment['confidence_std'],
            fmt='o',
            capsize=5
        )
        plt.title('Sentiment Confidence by Topic')
        plt.xlabel('Topic ID')
        plt.ylabel('Confidence Score (with std dev)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/topic_sentiment_confidence_{timestamp}.png")
        plt.close()
    
    def run_analysis(self, reviews: pd.Series, resume: bool = True) -> dict:
        """Enhanced run_analysis method with sentiment analysis"""
        # Run the original analysis
        results = super().run_analysis(reviews, resume)
        
        # Add sentiment analysis
        logger.info("Performing sentiment analysis...")
        topic_sentiment = self._analyze_topic_sentiment(
            results['sentences'],
            results['results']['topics']
        )
        
        # Add sentiment results to the final results
        results['sentiment_analysis'] = topic_sentiment.to_dict(orient='records')
        
        # Create sentiment visualizations
        base_dir = self.checkpoint_dir.split('_checkpoints')[0]
        viz_dir = f"{base_dir}_visualizations" if '_checkpoints' in self.checkpoint_dir else 'visualizations'
        self.create_sentiment_visualizations(topic_sentiment, viz_dir)
        
        # Update the summary file with sentiment information
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = f"{base_dir}_results" if '_checkpoints' in self.checkpoint_dir else 'results'
        
        with open(f"{results_dir}/sentiment_analysis_{timestamp}.txt", 'w') as f:
            f.write("Topic Sentiment Analysis Summary\n")
            f.write("=============================\n\n")
            
            # Sort topics by absolute sentiment for reporting
            topic_sentiment['abs_sentiment'] = abs(topic_sentiment['sentiment_mean'])
            sorted_sentiment = topic_sentiment.sort_values('abs_sentiment', ascending=False)
            
            for _, row in sorted_sentiment.iterrows():
                sentiment_direction = "POSITIVE" if row['sentiment_mean'] > 0 else "NEGATIVE"
                f.write(f"Topic {row['topic']}:\n")
                f.write(f"  Average Sentiment: {row['sentiment_mean']:.3f} ({sentiment_direction})\n")
                f.write(f"  Confidence: {row['confidence_mean']:.3f} ± {row['confidence_std']:.3f}\n")
                f.write(f"  Number of Sentences: {row['sentence_count']}\n\n")
        
        return results