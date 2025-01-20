from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for the analysis pipeline"""
    sentence_transformer_model: str = 'all-MiniLM-L6-v2'
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.1
    hdbscan_min_cluster_size: int = 15
    bertopic_model: str = "all-MiniLM-L6-v2"
