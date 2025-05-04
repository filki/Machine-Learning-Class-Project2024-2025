import os
import pandas as pd
import json
import glob
from datetime import datetime
import logging
from src.ml.analyzer import SteamReviewAnalyzer
from src.ml.utils import save_analysis_results

logger = logging.getLogger(__name__)

class DataService:
    """Service for handling data operations"""
    
    def __init__(self):
        self.data_path = 'data/downloader/rpg_100/dataset_combined.csv'
        self.results_dir = os.path.join('output', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def get_stats(self):
        """Get statistics about the dataset"""
        try:
            # Load the dataset
            df = pd.read_csv(self.data_path)
            
            # Count unique games and total reviews
            game_count = df['app_id'].nunique() if 'app_id' in df.columns else 0
            review_count = len(df)
            
            # Get other interesting stats
            positive_reviews = df[df['review_score'] > 0].shape[0] if 'review_score' in df.columns else 0
            negative_reviews = df[df['review_score'] <= 0].shape[0] if 'review_score' in df.columns else 0
            
            # Calculate positive percentage
            positive_percentage = (positive_reviews / review_count * 100) if review_count > 0 else 0
            
            return {
                'game_count': game_count,
                'review_count': review_count,
                'positive_reviews': positive_reviews,
                'negative_reviews': negative_reviews,
                'positive_percentage': round(positive_percentage, 2)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                'game_count': 0,
                'review_count': 0,
                'positive_reviews': 0,
                'negative_reviews': 0,
                'positive_percentage': 0
            }
    
    def get_recent_analyses(self):
        """Get list of recent analysis results"""
        try:
            # Get all JSON files in the results directory
            result_files = glob.glob(os.path.join(self.results_dir, '*.json'))
            
            # Sort by modification time (newest first)
            result_files.sort(key=os.path.getmtime, reverse=True)
            
            # Get basic info about each file
            analyses = []
            for file_path in result_files[:5]:  # Show only 5 most recent
                filename = os.path.basename(file_path)
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Try to get some basic info from the file
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        topic_count = len(data.get('sentiment_analysis', []))
                except:
                    topic_count = "Unknown"
                
                analyses.append({
                    'filename': filename,
                    'date': mod_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'topic_count': topic_count
                })
            
            return analyses
        except Exception as e:
            logger.error(f"Error getting recent analyses: {str(e)}")
            return []
    
    def run_analysis(self, sample_size=2500):
        """Run the analysis on the dataset"""
        try:
            # Load data
            df = pd.read_csv(self.data_path)
            
            # Sample if requested
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size)
            
            # Run analysis
            analyzer = SteamReviewAnalyzer()
            results = analyzer.run_analysis(df['review'])
            
            # Save results
            results_file = save_analysis_results(results, self.results_dir)
            
            return results_file
        except Exception as e:
            logger.error(f"Error running analysis: {str(e)}")
            raise
    
    def load_results(self, filename):
        """Load a specific results file"""
        try:
            file_path = os.path.join(self.results_dir, filename)
            
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            return results
        except Exception as e:
            logger.error(f"Error loading results file: {str(e)}")
            raise