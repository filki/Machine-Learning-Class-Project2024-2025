from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import LiteLLM, OpenAI, BaseRepresentation
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import hdbscan
import torch
from .config import UMAP_CONFIG, HDBSCAN_CONFIG, TRANSFORMER_MODEL, VECTORIZER_CONFIG
from dotenv import load_dotenv
import os


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

    load_dotenv(override=True)
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    representation_model = LiteLLM(model='gemini/gemini-2.0-flash-lite', delay_in_seconds=2.5)

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=CountVectorizer(**VECTORIZER_CONFIG),
        representation_model=representation_model,
        min_topic_size=500,
        verbose=True
    )