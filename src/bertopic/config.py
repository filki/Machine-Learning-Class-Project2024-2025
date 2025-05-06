from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'

CUSTOM_STOP_WORDS = [
    'game', 'games', 'play', 'played', 'playing',
    'steam', 'review', 'reviews', 'recommend',
    'hour', 'hours']

COMBINED_STOP_WORDS = list(set(ENGLISH_STOP_WORDS).union(CUSTOM_STOP_WORDS))

UMAP_CONFIG = {
    'n_neighbors': 30,
    'n_components': 10,
    'metric': 'cosine',
    'min_dist': 0.1,
    'random_state': 42
}
HDBSCAN_CONFIG = {
    'min_cluster_size': 500,
    'min_samples': 100,
    'metric': 'euclidean',
    'prediction_data': True,
    'gen_min_span_tree': True,
    'cluster_selection_method': 'eom',
    'cluster_selection_epsilon': 0.3
}
VECTORIZER_CONFIG = {
    'min_df': 2,
    'max_df': 0.7,
    'ngram_range': (1, 2),
    'stop_words': COMBINED_STOP_WORDS
}