import os
import json
import pickle
from datetime import datetime
import hashlib
import logging
from typing import Any, Tuple
import blingfire
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd

def generate_checkpoint_id(data) -> str:
    """Generate a unique identifier for the dataset"""
    sample = str(data.head(3).values) + str(data.shape)
    return hashlib.md5(sample.encode()).hexdigest()[:10]

def save_checkpoint(checkpoint_dir: str, stage: str, data: Any, checkpoint_id: str):
    """Save checkpoint data to file"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{checkpoint_dir}/{checkpoint_id}_{stage}_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'stage': stage,
        'filename': filename
    }
    
    metadata_file = f"{checkpoint_dir}/{checkpoint_id}_metadata.json"
    with open(metadata_file, 'a') as f:
        json.dump(metadata, f)
        f.write('\n')

def save_analysis_results(results: dict, results_dir: str) -> str:
    """Save analysis results with appropriate serialization per type
    
    Args:
        results: Analysis results dictionary containing topics, clusters, etc.
        results_dir: Directory to save results in
        
    Returns:
        str: Path to the saved pickle file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Save full results as pickle for complete data preservation
        pickle_path = os.path.join(results_dir, f'full_results_{timestamp}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
            
        # Save a JSON-friendly summary
        summary = {
            'timestamp': timestamp,
            'num_sentences': len(results['sentences']),
            'checkpoint_id': results['checkpoint_id']
        }
        
        if 'results' in results:
            # Save topic information
            topic_model = results['results']['topic_model']
            topic_info = topic_model.get_topic_info()
            topic_info.to_csv(os.path.join(results_dir, f'topic_info_{timestamp}.csv'), index=False)
            
            # Save detailed topic words
            topic_details = []
            for topic_id in topic_info['Topic']:
                if topic_id != -1:  # Skip outlier topic
                    words = topic_model.get_topic(topic_id)
                    topic_details.append({
                        'topic_id': topic_id,
                        'words': [word for word, _ in words[:10]],  # Top 10 words
                        'scores': [score for _, score in words[:10]]
                    })
            pd.DataFrame(topic_details).to_csv(os.path.join(results_dir, f'topic_details_{timestamp}.csv'), index=False)
            
            # Save cluster information
            if 'clusters' in results['results']:
                cluster_info = pd.DataFrame({
                    'document_id': range(len(results['results']['clusters'])),
                    'cluster': results['results']['clusters'],
                    'topic': results['results']['topics']
                })
                cluster_info.to_csv(os.path.join(results_dir, f'cluster_assignments_{timestamp}.csv'), index=False)
                
                # Save cluster statistics
                cluster_stats = cluster_info.groupby('cluster').agg({
                    'document_id': 'count',
                    'topic': lambda x: len(set(x))
                }).rename(columns={
                    'document_id': 'size',
                    'topic': 'unique_topics'
                })
                cluster_stats.to_csv(os.path.join(results_dir, f'cluster_stats_{timestamp}.csv'))
            
            # Save document-topic mapping
            doc_topics = pd.DataFrame({
                'document_id': range(len(results['results']['topics'])),
                'topic': results['results']['topics'],
                'sentence': results['sentences']
            })
            doc_topics.to_csv(os.path.join(results_dir, f'document_topics_{timestamp}.csv'), index=False)
            
            # Save topic-topic similarity if available
            try:
                topic_sims = topic_model.topic_similarities()
                pd.DataFrame(topic_sims).to_csv(os.path.join(results_dir, f'topic_similarities_{timestamp}.csv'))
            except:
                logging.warning("Could not generate topic similarities")
                
            summary['topics'] = results['results']['topics'].tolist() if hasattr(results['results']['topics'], 'tolist') else results['results']['topics']
        
        # Save sentiment analysis if present
        if 'sentiment_analysis' in results:
            pd.DataFrame(results['sentiment_analysis']).to_csv(
                os.path.join(results_dir, f'sentiment_analysis_{timestamp}.csv'), 
                index=False
            )
            summary['sentiment_analysis'] = results['sentiment_analysis']
                
        # Save the JSON summary
        summary_path = os.path.join(results_dir, f'summary_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return pickle_path
    except Exception as e:
        logging.error(f"Error saving analysis results: {str(e)}")
        raise

def get_output_dirs(base_dir: str = 'output') -> tuple:
    """Create and return output directory paths
    
    Args:
        base_dir: Base directory for all outputs
        
    Returns:
        tuple: (checkpoints_dir, viz_dir, results_dir)
    """
    checkpoints_dir = os.path.join(base_dir, 'checkpoints')
    viz_dir = os.path.join(base_dir, 'visualizations')
    results_dir = os.path.join(base_dir, 'results')
    
    # Create all directories
    for directory in [base_dir, checkpoints_dir, viz_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)
        
    return checkpoints_dir, viz_dir, results_dir

def save_visualization(fig, filename: str, output_dir: str):
    """Save a matplotlib figure with proper directory handling"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    return path

def load_checkpoint(checkpoint_dir: str, stage: str, checkpoint_id: str) -> Tuple[Any, bool]:
    """Load the latest checkpoint for a given stage"""
    try:
        metadata_file = f"{checkpoint_dir}/{checkpoint_id}_metadata.json"
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
        
        with open(latest['filename'], 'rb') as f:
            data = pickle.load(f)
            
        return data, True
        
    except Exception as e:
        logging.warning(f"Failed to load checkpoint: {e}")
        return None, False

def preprocess_reviews(reviews):
    """Tokenize reviews into sentences"""
    reviews = reviews.dropna()
    reviews_tokenized = []
    
    for review in tqdm(reviews, desc="Preprocessing"):
        sentences = blingfire.text_to_sentences(review)
        reviews_tokenized.append(sentences)
    
    all_sentences = [
        sentence 
        for review in reviews_tokenized 
        for sentence in review.split('\n') 
        if sentence.strip()
    ]
    
    return all_sentences