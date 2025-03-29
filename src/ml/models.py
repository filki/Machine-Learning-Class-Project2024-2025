from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import hdbscan
import torch
from .config import UMAP_CONFIG, HDBSCAN_CONFIG, TRANSFORMER_MODEL, VECTORIZER_CONFIG

if torch.cuda.is_available():
    try:
        from cuml.manifold import UMAP as cuUMAP
        from cuml.cluster import HDBSCAN as cuHDBSCAN
        use_gpu = True
    except ImportError:
        use_gpu = False
else:
    use_gpu = False


def get_transformer():
    device = 'cuda' if use_gpu else 'cpu'
    model = SentenceTransformer(TRANSFORMER_MODEL)
    return model.to(device)

def get_bertopic():
    if use_gpu:
        umap_model = cuUMAP(**UMAP_CONFIG)
        hdbscan_model = cuHDBSCAN(**HDBSCAN_CONFIG)
    else:
        umap_model = UMAP(**UMAP_CONFIG)
        hdbscan_model = hdbscan.HDBSCAN(**HDBSCAN_CONFIG)

    embedding_model = TRANSFORMER_MODEL

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=CountVectorizer(**VECTORIZER_CONFIG),
        min_topic_size=500,
        verbose=True
    )

