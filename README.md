# Machine-Learning-Class-Project2024-2025

## Steam Review Analysis POC

This project performs topic modeling and clustering on Steam video game reviews to extract insights and visualize review topics.

## Requirements

- Python 3.10 (required due to dependencies)
- dependencies in requirements.txt

## Usage

To run the complete analysis pipeline:

```
    python test_pipeline.py
```

This will:
1. Test the checkpointing functionality.
2. Test the analysis on a small sample of real review data.
3. Save visualizations to the `visualizations/` directory.
4. Save detailed analysis results to the `results/` directory.

## Configuration

The `ModelConfig` dataclass in `steam_analysis.py` contains configurable parameters for the analysis models:

- `sentence_transformer_model`: SentenceTransformer model for embedding creation ('all-MiniLM-L6-v2' used as a very small free access model for the POC, better models to be checked in the future e.g 'stella_en_1.5B_v5')
- `umap_n_neighbors`: Number of neighbors for UMAP dimensionality reduction (default: 15)
- `umap_n_components`: Number of components for UMAP reduction (default: 5) 
- `hdbscan_min_cluster_size`: Minimum cluster size for HDBSCAN clustering (default: 15)
- `bertopic_model`: BERTopic model for topic modeling ('all-MiniLM-L6-v2' - same as above)

## Results

The analysis pipeline produces the following outputs:

- `visualizations/`: Visualizations of topic sizes, word clouds, clusters, and topic-cluster relationships.
- `results/`: Detailed analysis results including full result pickle file, topic info CSV, cluster stats CSV, and summary text file.




