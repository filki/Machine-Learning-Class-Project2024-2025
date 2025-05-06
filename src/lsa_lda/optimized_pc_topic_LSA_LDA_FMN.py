import os
import pandas as pd
import numpy as np
from time import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from umap.umap_ import UMAP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, NMF
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import psutil
from joblib import Memory
import tempfile

CACHE_DIR = os.path.join(tempfile.gettempdir(), 'topic_modeling_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(location=CACHE_DIR, verbose=0)

def print_resource_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"CPU Usage: {cpu_percent}%")
    print(f"RAM Usage: {memory.percent}%")


def download_nltk_resources():
    try:
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
    except ImportError:
        pass

    resources = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger'
    ]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Couldn't download {resource}: {str(e)}")
            if resource == 'punkt_tab':
                # Fallback do punkt jeśli punkt_tab nie jest dostępny
                nltk.download('punkt', quiet=True)


def get_custom_stop_words():
    gaming_stop_words = {
        'game', 'games', 'play', 'played', 'playing',
        'steam', 'review', 'reviews', 'recommend',
        'recommended', 'hour', 'hours', 'like', 'good',
        'bad', 'great', 'best', 'worst', 'really',
        'much', 'many', 'lot', 'well', 'make', 'made'
    }
    return list(set(stopwords.words('english')).union(gaming_stop_words))


@memory.cache
def process_text(text, lemmatizer=None, stop_words=None):
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    if stop_words is None:
        stop_words = get_custom_stop_words()

    if not isinstance(text, str):
        return []

    processed_sentences = []
    for sent in sent_tokenize(text.lower()):
        sent = re.sub(r'http\S+|www\S+|[^\w\s]', ' ', sent)
        tokens = word_tokenize(sent)
        tokens = [lemmatizer.lemmatize(word) for word in tokens
                  if word not in stop_words and len(word) > 2]
        if tokens:
            processed_sentences.append(' '.join(tokens))

    return processed_sentences


def process_text_chunk(chunk, lemmatizer=None, stop_words=None):
    all_sentences = []
    for text in chunk:
        sentences = process_text(text, lemmatizer, stop_words)
        all_sentences.extend(sentences)
    return all_sentences


def preprocess_data_parallel(data, n_cores=None):
    if n_cores is None:
        n_cores = 14  # Ryzen 7 5800X - zostawiamy 2 wątki na system

    chunk_size = max(1, len(data) // (n_cores * 2))  # mniejsze chunki dla lepszego zrównoleglenia
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    print(f"Using {n_cores} cores for processing {len(chunks)} chunks")
    print(f"Chunk size: {chunk_size} documents")

    lemmatizer = WordNetLemmatizer()
    stop_words = get_custom_stop_words()

    processed_sentences = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        process_func = partial(process_text_chunk,
                               lemmatizer=lemmatizer,
                               stop_words=stop_words)
        results = list(tqdm(executor.map(process_func, chunks),
                            total=len(chunks),
                            desc="Processing chunks"))

    for chunk_sentences in results:
        processed_sentences.extend(chunk_sentences)

    print(f"Total sentences extracted: {len(processed_sentences)}")
    return processed_sentences


@memory.cache
def calculate_coherence(model, vectorizer, X, top_n=10):
    if isinstance(model, (TruncatedSVD, NMF)):
        components = model.components_
    else:  # LDA
        components = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]

    feature_names = vectorizer.get_feature_names_out()
    coherence_scores = []

    for topic in components:
        top_terms_idx = topic.argsort()[:-top_n - 1:-1]
        term_vectors = X[:, top_terms_idx].toarray()
        term_similarities = cosine_similarity(term_vectors.T)
        coherence = (term_similarities.sum() - term_similarities.shape[0]) / \
                    (term_similarities.shape[0] * (term_similarities.shape[0] - 1))
        coherence_scores.append(coherence)

    return np.mean(coherence_scores)


@memory.cache
def calculate_diversity(model, vectorizer, top_n=10):
    if isinstance(model, (TruncatedSVD, NMF)):
        components = model.components_
    else:  # LDA
        components = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]

    feature_names = vectorizer.get_feature_names_out()
    all_top_words = set()

    for topic in components:
        top_terms_idx = topic.argsort()[:-top_n - 1:-1]
        top_words = {feature_names[i] for i in top_terms_idx}
        all_top_words.update(top_words)

    diversity = len(all_top_words) / (model.n_components * top_n)
    return diversity


@memory.cache
def calculate_stability(model, X, vectorizer, n_samples=5):
    stability_scores = []

    for _ in range(n_samples):
        indices = np.random.choice(X.shape[0], size=int(X.shape[0] * 0.8), replace=False)
        X_sample = X[indices]

        if isinstance(model, TruncatedSVD):
            model_sample = TruncatedSVD(n_components=model.n_components, random_state=42)
        elif isinstance(model, LatentDirichletAllocation):
            model_sample = LatentDirichletAllocation(
                n_components=model.n_components,
                random_state=42,
                n_jobs=14  # Zoptymalizowane dla Ryzen 7
            )
        else:  # NMF
            model_sample = NMF(n_components=model.n_components, random_state=42)

        model_sample.fit(X_sample)

        if isinstance(model, (TruncatedSVD, NMF)):
            similarity = cosine_similarity(model.components_, model_sample.components_)
        else:  # LDA
            components1 = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
            components2 = model_sample.components_ / model_sample.components_.sum(axis=1)[:, np.newaxis]
            similarity = cosine_similarity(components1, components2)

        stability_scores.append(np.mean(np.max(similarity, axis=1)))

    return np.mean(stability_scores)


def visualize_topics(model, X, method_name, save_path):
    print("Performing document transformation...")
    doc_topics = model.transform(X)

    max_samples = 50000
    if doc_topics.shape[0] > max_samples:
        print(f"Sampling {max_samples} documents for visualization...")
        indices = np.random.choice(doc_topics.shape[0], max_samples, replace=False)
        doc_topics_sampled = doc_topics[indices]
    else:
        doc_topics_sampled = doc_topics

    print("Running UMAP dimensionality reduction...")
    try:
        umap = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42,
            n_jobs=14,
            verbose=True
        )
        umap_output = umap.fit_transform(doc_topics_sampled)
    except Exception as e:
        print(f"UMAP failed with {max_samples} samples: {e}")
        max_samples = max_samples // 2
        indices = np.random.choice(doc_topics.shape[0], max_samples, replace=False)
        doc_topics_sampled = doc_topics[indices]
        umap_output = umap.fit_transform(doc_topics_sampled)

    dominant_topics = np.argmax(doc_topics_sampled, axis=1)

    plt.figure(figsize=(16, 10))

    # Remove seaborn style and use a built-in style instead
    plt.style.use('default')

    scatter = plt.scatter(umap_output[:, 0],
                          umap_output[:, 1],
                          c=dominant_topics,
                          cmap='tab20',
                          alpha=0.6,
                          s=15)

    plt.colorbar(scatter, label='Topic Number')
    plt.title(f'{method_name} Topic Distribution (UMAP visualization)',
              pad=20,
              fontsize=14,
              fontweight='bold')
    plt.xlabel('UMAP dimension 1', fontsize=12)
    plt.ylabel('UMAP dimension 2', fontsize=12)

    info_text = f'Total topics: {model.n_components}\n'
    info_text += f'Samples visualized: {doc_topics_sampled.shape[0]}'
    plt.figtext(0.02, 0.02, info_text, fontsize=8, alpha=0.7)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {save_path}")


@memory.cache
def find_optimal_topics(texts, method, min_topics, max_topics, step):
    if method == 'lda':
        vectorizer = CountVectorizer(
            max_features=12000,
            min_df=10,
            max_df=0.75,
            stop_words=get_custom_stop_words(),
            dtype=np.float32  # Zmniejszenie zużycia pamięci
        )
    else:  # LSA lub NMF
        vectorizer = TfidfVectorizer(
            max_features=12000,
            min_df=10,
            max_df=0.75,
            stop_words=get_custom_stop_words(),
            dtype=np.float32  # Zmniejszenie zużycia pamięci
        )

    X = vectorizer.fit_transform(texts)

    total_sentences = X.shape[0]
    min_cluster_size = int(total_sentences * 0.0005)
    max_cluster_size = int(total_sentences * 0.02)

    print(f"Total sentences: {total_sentences}")
    print(f"Min sentences per topic: {min_cluster_size}")
    print(f"Max sentences per topic: {max_cluster_size}")

    scores = []
    best_score = float('-inf')
    best_config = None

    print(f"\nFinding optimal number of topics for {method.upper()}...")
    for n_topics in tqdm(range(min_topics, max_topics + 1, step)):
        if method == 'lsa':
            model = TruncatedSVD(n_components=n_topics, random_state=42)
        elif method == 'lda':
            model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                n_jobs=14,  # Zoptymalizowane dla Ryzen 7
                batch_size=256,
                max_iter=20,
                evaluate_every=2
            )
        else:  # nmf
            model = NMF(
                n_components=n_topics,
                random_state=42,
                init='nndsvd',  # Lepsza inicjalizacja
                solver='cd',  # Szybszy solver
                batch_size=256
            )

        model.fit(X)

        if isinstance(model, (TruncatedSVD, NMF)):
            doc_topic_dist = model.transform(X)
        else:  # LDA
            doc_topic_dist = model.transform(X)

        doc_topics = np.argmax(doc_topic_dist, axis=1)
        topic_sizes = np.bincount(doc_topics, minlength=n_topics)

        valid_topics = np.sum((topic_sizes >= min_cluster_size) &
                              (topic_sizes <= max_cluster_size))
        if valid_topics < n_topics * 0.8:
            print(f"Warning: Only {valid_topics}/{n_topics} topics meet size criteria")
            continue

        coherence = calculate_coherence(model, vectorizer, X)
        diversity = calculate_diversity(model, vectorizer)
        stability = calculate_stability(model, X, vectorizer)

        if method == 'lsa':
            performance = model.explained_variance_ratio_.sum()
        elif method == 'lda':
            performance = model.score(X)
        else:  # nmf
            performance = -model.reconstruction_err_

        norm_coherence = coherence / (1 + coherence)
        norm_diversity = diversity
        norm_stability = stability
        norm_performance = (performance - min(performance, 0)) / (1 + abs(performance))

        weights = {
            'coherence': 0.3,
            'diversity': 0.2,
            'stability': 0.2,
            'performance': 0.3
        }

        combined_score = (
                weights['coherence'] * norm_coherence +
                weights['diversity'] * norm_diversity +
                weights['stability'] * norm_stability +
                weights['performance'] * norm_performance
        )

        scores.append((n_topics, coherence, diversity, stability, performance,
                       combined_score, valid_topics))

        if combined_score > best_score:
            best_score = combined_score
            best_config = {
                'n_topics': n_topics,
                'model': model,
                'vectorizer': vectorizer,
                'coherence': coherence,
                'diversity': diversity,
                'stability': stability,
                'performance': performance,
                'score': combined_score,
                'valid_topics': valid_topics
            }

        print(
            f"Topics: {n_topics}, "
            f"Valid topics: {valid_topics}/{n_topics}, "
            f"Coherence: {coherence:.4f}, "
            f"Diversity: {diversity:.4f}, "
            f"Stability: {stability:.4f}, "
            f"Performance: {performance:.4f}, "
            f"Score: {combined_score:.4f}"
        )
        gc.collect()

    return scores, best_config

def save_results(config, method, log_file_path, original_texts):
    model = config['model']
    vectorizer = config['vectorizer']
    feature_names = vectorizer.get_feature_names_out()

    results = f"\nNumber of topics: {model.n_components}\n"
    results += f"Coherence: {config['coherence']:.4f}\n"
    results += f"Diversity: {config['diversity']:.4f}\n"
    results += f"Stability: {config['stability']:.4f}\n"
    results += f"Performance: {config['performance']:.4f}\n"
    results += f"Combined Score: {config['score']:.4f}\n\n"

    if isinstance(model, (TruncatedSVD, NMF)):
        components = model.components_
        if isinstance(model, TruncatedSVD):
            topic_importance = np.abs(model.explained_variance_ratio_)
        else:  # NMF
            topic_importance = np.sum(components, axis=1)
    else:  # LDA
        components = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
        topic_importance = np.sum(components, axis=1)

    topic_order = np.argsort(-topic_importance)

    for idx, topic_idx in enumerate(topic_order):
        top_words_idx = components[topic_idx].argsort()[:-20:-1]
        top_words = [(feature_names[i], components[topic_idx][i])
                     for i in top_words_idx]

        results += f"Topic {idx + 1} (Importance: {topic_importance[topic_idx]:.4f}):\n"
        for word, weight in top_words:
            results += f"  - {word}: {weight:.4f}\n"
        results += "\n"

    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(results)

    viz_path = log_file_path.replace('.txt', '_visualization.png')
    print("\nGenerating topic visualization...")
    visualize_topics(model, vectorizer.transform(original_texts), method.upper(), viz_path)

def main():
    data_path = '../../data/dataset_combined.csv'
    results_path_lsa = '../../logs/sklearn_LSA_results.txt'
    results_path_lda = '../../logs/sklearn_LDA_results.txt'
    results_path_nmf = '../../logs/sklearn_NMF_results.txt'
    min_topics, max_topics, step = 75, 200, 25

    print("\nInitial resource usage:")
    print_resource_usage()

    os.makedirs(os.path.dirname(results_path_lsa), exist_ok=True)
    download_nltk_resources()

    print("Loading data...")
    data = pd.read_csv(data_path, low_memory=False)

    print("\nPreprocessing texts...")
    texts = preprocess_data_parallel(data['review'])

    print("\nResource usage after preprocessing:")
    print_resource_usage()

    # Analiza LSA
    print("\nResource usage before LSA:")
    print_resource_usage()

    print("\nPerforming LSA analysis...")
    t0 = time()
    lsa_scores, lsa_best = find_optimal_topics(
        texts,
        method='lsa',
        min_topics=min_topics,
        max_topics=max_topics,
        step=step
    )
    print("\nSaving LSA results...")
    save_results(lsa_best, 'LSA', results_path_lsa, texts)
    print(f"LSA analysis completed in {time() - t0:.2f} seconds")

    print("\nResource usage after LSA:")
    print_resource_usage()

    # Analiza LDA
    print("\nResource usage before LDA:")
    print_resource_usage()

    print("\nPerforming LDA analysis...")
    t0 = time()
    lda_scores, lda_best = find_optimal_topics(
        texts,
        method='lda',
        min_topics=min_topics,
        max_topics=max_topics,
        step=step
    )
    print("\nSaving LDA results...")
    save_results(lda_best, 'LDA', results_path_lda, texts)
    print(f"LDA analysis completed in {time() - t0:.2f} seconds")

    print("\nResource usage after LDA:")
    print_resource_usage()

    # Analiza NMF
    print("\nResource usage before NMF:")
    print_resource_usage()

    print("\nPerforming NMF analysis...")
    t0 = time()
    nmf_scores, nmf_best = find_optimal_topics(
        texts,
        method='nmf',
        min_topics=min_topics,
        max_topics=max_topics,
        step=step
    )
    print("\nSaving NMF results...")
    save_results(nmf_best, 'NMF', results_path_nmf, texts)
    print(f"NMF analysis completed in {time() - t0:.2f} seconds")

    print("\nFinal resource usage:")
    print_resource_usage()

if __name__ == "__main__":
    main()