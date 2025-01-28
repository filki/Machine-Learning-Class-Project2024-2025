import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Optional
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_plot(fig, output_dir: Path, filename: str) -> str:
    filepath = output_dir / filename
    if hasattr(fig, 'write_html'):
        fig.write_html(filepath)
    elif hasattr(fig, 'savefig'):
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
    else:
        raise ValueError("Invalid figure object provided")
    return str(filepath)

def create_topic_size_plot(topic_info: pd.DataFrame, output_dir: Path, timestamp: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    top_topics = topic_info.head(10)
    sns.barplot(data=top_topics, x='Topic', y='Count', ax=ax)
    ax.set_title('Top 10 Topics by Size')
    ax.set_xlabel('Topic ID')
    ax.set_ylabel('Number of Documents')
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = save_plot(fig, output_dir, f'topic_sizes_{timestamp}.png')
    plt.close(fig)
    return path

def create_wordclouds(topic_model, top_topics: pd.DataFrame, output_dir: Path, timestamp: str) -> Dict[str, str]:
    wordcloud_paths = {}
    try:
        for topic_id in top_topics['Topic'][:5]:
            if topic_id != -1:
                words = dict(topic_model.get_topic(topic_id))
                if words:
                    wordcloud = WordCloud(
                        width=800, height=400,
                        background_color='white',
                        stopwords=STOPWORDS.union(ENGLISH_STOP_WORDS),
                        min_word_length=3,
                        collocations=True
                    ).generate_from_frequencies(words)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f'Topic {topic_id} Word Cloud')
                    filepath = save_plot(fig, output_dir, f'topic_{topic_id}_wordcloud_{timestamp}.png')
                    wordcloud_paths[f'wordcloud_topic_{topic_id}'] = filepath
                    plt.close(fig)
    except Exception as e:
        logger.warning(f"Error creating wordclouds: {str(e)}")
    return wordcloud_paths

def create_topic_similarity_matrix(topic_model, output_dir: Path, timestamp: str) -> Optional[str]:
    try:
        topic_embeddings = topic_model.topic_embeddings_
        if topic_embeddings is None:
            return None
        
        similarities = cosine_similarity(topic_embeddings)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            pd.DataFrame(similarities),
            cmap='YlOrRd',
            center=0,
            square=True,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Similarity Score'},
            ax=ax
        )
        ax.set_title('Topic Similarity Matrix')
        plt.tight_layout()
        path = save_plot(fig, output_dir, f'topic_similarities_{timestamp}.png')
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"Could not create topic similarity visualization: {str(e)}")
        return None

def create_sentiment_visualizations(topic_sentiment: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sentiment_viz_paths = {}
    
    try:
        topic_sentiment['topic'] = pd.to_numeric(topic_sentiment['topic'])
        
        # Topic Sentiment Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=topic_sentiment, x='topic', y='sentiment_mean', color='skyblue', ax=ax)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax.set_title('Average Sentiment by Topic')
        ax.set_xlabel('Topic ID')
        ax.set_ylabel('Average Sentiment (-1 to 1)')
        plt.xticks(rotation=45)
        sentiment_viz_paths['sentiment_distribution'] = save_plot(
            fig, output_dir, f'topic_sentiment_{timestamp}.png'
        )
        plt.close(fig)
        
        # Sentiment Confidence Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.errorbar(
            topic_sentiment['topic'],
            topic_sentiment['confidence_mean'],
            yerr=topic_sentiment['confidence_std'],
            fmt='o',
            capsize=5,
            color='skyblue',
            ecolor='gray'
        )
        ax.set_title('Sentiment Classification Confidence by Topic')
        ax.set_xlabel('Topic ID')
        ax.set_ylabel('Average Confidence Score (with std dev)')
        ax.grid(True, alpha=0.3)
        sentiment_viz_paths['sentiment_confidence'] = save_plot(
            fig, output_dir, f'topic_sentiment_confidence_{timestamp}.png'
        )
        plt.close(fig)
        
        # Sentiment Volume Distribution
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=topic_sentiment, x='topic', y='sentence_count', color='lightgreen', ax=ax)
        ax.set_title('Number of Sentences per Topic')
        ax.set_xlabel('Topic ID')
        ax.set_ylabel('Number of Sentences')
        plt.xticks(rotation=45)
        sentiment_viz_paths['sentiment_volume'] = save_plot(
            fig, output_dir, f'topic_sentiment_volume_{timestamp}.png'
        )
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Error creating sentiment visualizations: {str(e)}")
    
    return sentiment_viz_paths

def create_bertopic_visualizations(topic_model, output_dir: Path, timestamp: str) -> Dict[str, str]:
    viz_paths = {}
    try:
        # Topic visualization
        topics_df = topic_model.get_topic_info()
        if len(topics_df) > 1:
            topics_fig = topic_model.visualize_topics()
            viz_paths['topics'] = save_plot(topics_fig, output_dir, f'topic_visualization_{timestamp}.html')
        
        # Topic terms visualization
        terms_fig = topic_model.visualize_barchart()
        viz_paths['terms'] = save_plot(terms_fig, output_dir, f'topic_terms_{timestamp}.html')
        
    except Exception as e:
        logger.warning(f"Error creating BERTopic visualizations: {str(e)}")
    
    return viz_paths

def create_visualizations(results: dict, output_dir: str = 'visualizations') -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_visualizations = {}
    
    try:
        if 'results' not in results or 'topic_model' not in results['results']:
            logger.warning("No topic model found in results")
            return saved_visualizations
            
        topic_model = results['results']['topic_model']
        topic_info = topic_model.get_topic_info()
        topic_info['Topic'] = pd.to_numeric(topic_info['Topic'])
        
        # Create visualizations
        saved_visualizations['topic_sizes'] = create_topic_size_plot(topic_info, output_dir, timestamp)
        saved_visualizations.update(create_wordclouds(topic_model, topic_info, output_dir, timestamp))
        
        similarity_path = create_topic_similarity_matrix(topic_model, output_dir, timestamp)
        if similarity_path:
            saved_visualizations['similarity_matrix'] = similarity_path
        
        saved_visualizations.update(create_bertopic_visualizations(topic_model, output_dir, timestamp))
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'visualizations': saved_visualizations,
            'parameters': {
                'num_topics': len(topic_info),
                'num_documents': len(results.get('sentences', [])),
            }
        }
        
        metadata_path = output_dir / f'visualization_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_visualizations['metadata'] = str(metadata_path)
        
        return saved_visualizations
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        raise