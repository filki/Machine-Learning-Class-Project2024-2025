import pandas as pd
import logging
from pathlib import Path
import shutil
import os
from steam_analysis import SteamReviewAnalyzer, ModelConfig, EnhancedSteamReviewAnalyzer
import re


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_tiny_dataset():
    """Create a very small test dataset"""
    return pd.Series([
        "This game is amazing! The graphics are beautiful and the story is engaging.",
        "Terrible optimization. Keeps crashing on my PC. Would not recommend.",
        "Great multiplayer experience, but single player needs work.",
        "Average game, nothing special. Decent graphics but boring gameplay.",
        "One of the best games I've played this year! Highly recommend it!"
    ])

def verify_outputs(base_dir: str, timestamp_pattern: str) -> bool:
    """Verify that all expected output files were created"""
    import re
    
    logger.info(f"Verifying outputs in directory: {base_dir}")
    
    if not os.path.exists(base_dir):
        logger.error(f"Directory does not exist: {base_dir}")
        return False
        
    files = os.listdir(base_dir)
    if not files:
        logger.error(f"No files found in directory: {base_dir}")
        return False
        
    logger.info(f"Found {len(files)} files in {base_dir}")
    
    # Different patterns for different directories
    if 'visualizations' in base_dir:
        essential_patterns = [
            r"^topic_sizes_.*\.png$",
            r"^clusters_.*\.png$",
            r"^topic_cluster_relationship_.*\.png$"
        ]
    else:  # results directory
        essential_patterns = [
            r"^topic_info_.*\.csv$",
            r"^topic_details_.*\.csv$",
            r"^cluster_stats_.*\.csv$",
            r"^analysis_summary_.*\.txt$"
        ]
    
    # Check for essential files
    for pattern in essential_patterns:
        matching_files = [f for f in files if re.match(pattern, f)]
        if not matching_files:
            logger.error(f"Missing essential file with pattern: {pattern}")
            return False
        else:
            logger.info(f"Found file matching pattern '{pattern}': {matching_files[0]}")
    
    return True

def test_checkpointing():
    """Test the checkpointing functionality"""
    try:
        logger.info("Testing checkpointing functionality...")
        
        # Create test directories
        test_dirs = ['test_checkpoints', 'test_visualizations', 'test_results']
        for dir_name in test_dirs:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
            os.makedirs(dir_name)
        
        # Create test data
        test_reviews = create_tiny_dataset()
        
        # Initialize analyzer
        config = ModelConfig(
            sentence_transformer_model='all-MiniLM-L6-v2',
            umap_n_neighbors=2,
            umap_n_components=2,
            hdbscan_min_cluster_size=2,
            bertopic_model='all-MiniLM-L6-v2'
        )
        
        analyzer = SteamReviewAnalyzer(
            config=config,
            checkpoint_dir='test_checkpoints'
        )
        
        # First run - should create checkpoints
        logger.info("First run - creating checkpoints...")
        results1 = analyzer.run_analysis(test_reviews, resume=True)
        
        # Verify checkpoints were created
        checkpoint_id = results1['checkpoint_id']
        metadata_file = f"test_checkpoints/{checkpoint_id}_metadata.json"
        
        if not os.path.exists(metadata_file):
            logger.error("Checkpoint metadata file not created!")
            return False
        
        # Verify outputs were created
        if not verify_outputs('test_visualizations', r'\d{8}_\d{6}'):
            return False
            
        if not verify_outputs('test_results', r'\d{8}_\d{6}'):
            return False
            
        # Second run - should use checkpoints
        logger.info("Second run - should use existing checkpoints...")
        results2 = analyzer.run_analysis(test_reviews, resume=True)
        
        # Verify results are consistent
        if results1['checkpoint_id'] != results2['checkpoint_id']:
            logger.error("Checkpoint IDs don't match between runs!")
            return False
            
        logger.info("Checkpointing test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_with_real_data_sample():
    """Test the analyzer with a small sample of real data"""
    try:
        logger.info("Testing with real data sample...")
        
        data_file = 'data/dataset_combined.csv'
        if not Path(data_file).exists():
            logger.error(f"Data file not found: {data_file}")
            return False
            
        # Load and sample data
        df = pd.read_csv(data_file, low_memory=False)
        sample_df = df.sample(n=100)  # Reduced sample size for faster testing
        
        config = ModelConfig(
            sentence_transformer_model='all-MiniLM-L6-v2',
            umap_n_neighbors=2,
            umap_n_components=2,
            hdbscan_min_cluster_size=2,
            bertopic_model='all-MiniLM-L6-v2'
        )
        
        test_dirs = ['real_data_checkpoints', 'real_data_visualizations', 'real_data_results']
        for dir_name in test_dirs:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
            os.makedirs(dir_name)
        
        analyzer = EnhancedSteamReviewAnalyzer(
            config=config,
            checkpoint_dir='real_data_checkpoints'
        )
        
        logger.info("Running analysis...")
        results = analyzer.run_analysis(sample_df['review'], resume=True)
        
        if not results:
            logger.error("Analysis failed to produce results")
            return False
            
        logger.info("Analysis completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Test checkpointing functionality
    logger.info("\n=== Testing checkpointing ===")
    success1 = test_checkpointing()
    
    # Test with real data sample
    logger.info("\n=== Testing with real data sample ===")
    success2 = test_with_real_data_sample()
    
    # Exit with success only if both tests pass
    exit(0 if (success1 and success2) else 1)