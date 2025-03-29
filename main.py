import pandas as pd
import logging
from src.ml.analyzer import SteamReviewAnalyzer
from src.ml.utils import save_analysis_results
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Setup project structure
        results_dir = os.path.join('output', 'results')

        # Load data
        df = pd.read_csv('data/downloader/rpg/dataset_combined.csv')
        df = df.sample(n=500)  # for testing with sample
        
        # Run analysis
        analyzer = SteamReviewAnalyzer()
        results = analyzer.run_analysis(df['review'])
        
        # Save results
        try:
            results_file = save_analysis_results(results, results_dir)
            print(f"\nAnalysis completed successfully. Output locations:")
            print(f"Results directory: {results_dir}")
            print(f"Latest results file: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")

        # Try uploading results to GCP bucket
        try:
            output_base = "output"
            bucket_path = "gs://steam_reviews_test/output_final_4/"  
            os.system(f"gsutil -m cp -r {output_base} {bucket_path}")
            print(f"Results directory uploaded to {bucket_path}.")
        except Exception as e:
            logger.error(f"Failed to upload results to bucket: {str(e)}")    
        
    except Exception as e:
        logger.error(f"Analysis failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return

if __name__ == "__main__":
    main()
