import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_visualizations(results: dict, output_dir: str = 'visualizations'):
    """Create and save visualizations of the analysis results
    
    Args:
        results: Analysis results containing topic model and embeddings
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Topic Visualization
    logger.info("Creating topic visualization...")
    topic_model = results['results']['topic_model']
    
    # Get topic info and ensure numeric conversion
    topic_info = topic_model.get_topic_info()
    topic_info['Topic'] = pd.to_numeric(topic_info['Topic'])
    top_topics = topic_info.head(10)  # Get top 10 topics
    
    # Plot topic sizes
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_topics, x='Topic', y='Count')
    plt.title('Top 10 Topics by Size')
    plt.xlabel('Topic ID')
    plt.ylabel('Number of Documents')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/topic_sizes_{timestamp}.png")
    plt.close()
    
    # 2. Topic Word Cloud
    try:
        for topic_id in top_topics['Topic'][:5]:  # Top 5 topics
            words = dict(topic_model.get_topic(topic_id))
            if words:
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    stopwords=STOPWORDS.union(ENGLISH_STOP_WORDS),
                    min_word_length=3,
                    collocations=True
                ).generate_from_frequencies(words)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Topic {topic_id} Word Cloud')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/topic_{topic_id}_wordcloud_{timestamp}.png")
                plt.close()
    except ImportError:
        logger.warning("WordCloud package not installed. Skipping word clouds.")
    
    # 3. Topic Similarity Heatmap
    try:
        topic_sims = topic_model.topic_similarities()
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pd.DataFrame(topic_sims),
            cmap='YlOrRd',
            center=0,
            square=True
        )
        plt.title('Topic Similarity Matrix')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/topic_similarities_{timestamp}.png")
        plt.close()
    except:
        logger.warning("Could not create topic similarity visualization")
    
    # 4. Topic-Document Distribution
    try:
        topic_distr = pd.DataFrame({
            'Topic': results['results']['topics'],
            'Count': 1
        }).groupby('Topic').count()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=topic_distr.reset_index(),
            x='Topic',
            y='Count',
            color='skyblue'
        )
        plt.title('Document Distribution Across Topics')
        plt.xlabel('Topic ID')
        plt.ylabel('Number of Documents')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/topic_distribution_{timestamp}.png")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create topic distribution plot: {e}")
    
    return f"{output_dir}/results_{timestamp}"

def create_sentiment_visualizations(topic_sentiment: pd.DataFrame, output_dir: str):
    """Create visualizations for sentiment analysis results
    
    Args:
        topic_sentiment: DataFrame with topic sentiment analysis results
        output_dir: Directory to save visualizations
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Topic Sentiment Distribution
    plt.figure(figsize=(12, 6))
    topic_sentiment['topic'] = pd.to_numeric(topic_sentiment['topic'])
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
    
    # 2. Sentiment Confidence Distribution
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        topic_sentiment['topic'],
        topic_sentiment['confidence_mean'],
        yerr=topic_sentiment['confidence_std'],
        fmt='o',
        capsize=5
    )
    plt.title('Sentiment Classification Confidence by Topic')
    plt.xlabel('Topic ID')
    plt.ylabel('Average Confidence Score (with std dev)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/topic_sentiment_confidence_{timestamp}.png")
    plt.close()
