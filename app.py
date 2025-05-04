from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import os
import logging
import json
from datetime import datetime
from src.ml.analyzer import SteamReviewAnalyzer
from src.ml.utils import save_analysis_results
from services.data_service import DataService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'steam_reviews_analysis_secret_key'

# Initialize data service
data_service = DataService()

@app.route('/')
def index():
    """Landing page showing stats and options"""
    try:
        # Get stats about games and reviews
        stats = data_service.get_stats()
        
        # Get list of recent analyses
        analyses = data_service.get_recent_analyses()
        
        return render_template('index.html', 
                              stats=stats,
                              analyses=analyses,
                              current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        logger.error(f"Error loading landing page: {str(e)}")
        flash(f"Error loading data: {str(e)}", "danger")
        return render_template('index.html', error=True)

@app.route('/run-analysis', methods=['POST'])
def run_analysis():
    """Run the analysis on the dataset"""
    try:
        # Get form parameters
        sample_size = request.form.get('sample_size', type=int, default=2500)
        
        # Run the analysis
        results_file = data_service.run_analysis(sample_size)
        
        flash(f"Analysis completed successfully! Results saved to {results_file}", "success")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        flash(f"Analysis failed: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/view-results/<filename>')
def view_results(filename):
    """View a specific analysis result"""
    try:
        # Load the results file
        results = data_service.load_results(filename)
        return render_template('results.html', results=results, filename=filename)
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        flash(f"Error loading results: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/api/stats')
def api_stats():
    """API endpoint for stats"""
    try:
        stats = data_service.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('output/results', exist_ok=True)
    app.run(debug=True)