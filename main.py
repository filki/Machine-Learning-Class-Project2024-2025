import pandas as pd
import logging
import os
from datetime import datetime
from src.ml.analyzer import SteamReviewAnalyzer
from src.ml.model_config import ModelConfig
from src.ml.utils import get_output_dirs, save_analysis_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Setup directory structure
        checkpoints_dir, viz_dir, results_dir = get_output_dirs('output')
        
        # Load data
        df = pd.read_csv('data/downloader/rpg_100/dataset_combined.csv')
        df = df.sample(n=500)  # for testing with sample
        
        # Initialize analyzer
        analyzer = SteamReviewAnalyzer(
            config=ModelConfig(),
            checkpoint_dir=checkpoints_dir,
            include_sentiment=True
        )
        
        # Run analysis
        results = analyzer.run_analysis(df['review'], viz_dir=viz_dir)
        
        # Save results
        try:
            results_file = save_analysis_results(results, results_dir)
            print(f"\nAnalysis completed successfully. Output locations:")
            print(f"- Checkpoints: {checkpoints_dir}")
            print(f"- Visualizations: {viz_dir}")
            print(f"- Results: {results_dir}")
            print(f"- Latest results file: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
        
    except Exception as e:
        logger.error(f"Analysis failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return

if __name__ == "__main__":
    main()