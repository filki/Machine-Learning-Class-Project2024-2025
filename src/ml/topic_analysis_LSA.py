import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
from time import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        nltk.download(resource, quiet=True)


def get_custom_stop_words():
    gaming_stop_words = {
        'game', 'games', 'play', 'played', 'playing',
        'steam', 'review', 'reviews', 'recommend',
        'recommended', 'hour', 'hours', 'like', 'good',
        'bad', 'great', 'best', 'worst', 'really',
        'much', 'many', 'lot', 'well', 'make', 'made'
    }
    return list(set(stopwords.words('english')).union(gaming_stop_words))


def process_text_chunk(chunk, lemmatizer=None, stop_words=None):
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    if stop_words is None:
        stop_words = get_custom_stop_words()

    sentences = []
    for text in chunk:
        if not isinstance(text, str):
            continue
        for sent in sent_tokenize(text.lower()):
            sent = re.sub(r'http\S+|www\S+|[^\w\s]', ' ', sent)
            tokens = word_tokenize(sent)
            tokens = [lemmatizer.lemmatize(word) for word in tokens
                      if word not in stop_words and len(word) > 2]
            if tokens:
                sentences.append(' '.join(tokens))
    return sentences


def preprocess_data_parallel(data, n_cores=None):
    if n_cores is None:
        n_cores = max(1, multiprocessing.cpu_count() - 2)

    chunk_size = max(1, len(data) // (n_cores * 4))
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


def calculate_coherence(svd, vectorizer, X, top_n=10):
    components = svd.components_
    coherence_scores = []

    for topic in components:
        top_terms_idx = topic.argsort()[:-top_n - 1:-1]
        term_vectors = X[:, top_terms_idx].toarray()
        term_similarities = cosine_similarity(term_vectors.T)
        coherence = (term_similarities.sum() - term_similarities.shape[0]) / \
                    (term_similarities.shape[0] * (term_similarities.shape[0] - 1))
        coherence_scores.append(coherence)

    return np.mean(coherence_scores)


def calculate_diversity(svd, vectorizer, top_n=10):
    """
    Oblicza różnorodność tematów na podstawie unikalnych słów.
    """
    components = svd.components_
    feature_names = vectorizer.get_feature_names_out()
    all_top_words = set()

    for topic in components:
        top_term_indices = topic.argsort()[:-top_n - 1:-1]
        top_words = {feature_names[i] for i in top_term_indices}
        all_top_words.update(top_words)

    # Normalizacja przez maksymalną możliwą liczbę unikalnych słów
    diversity = len(all_top_words) / (top_n * len(components))
    return diversity


def calculate_topic_stability(svd, X, n_samples=5):
    """
    Oblicza stabilność tematów poprzez porównanie wyników na różnych podzbiorach danych.
    """
    doc_topic_dist = svd.transform(X)
    stability_scores = []

    for _ in range(n_samples):
        # Losowy podzbiór dokumentów (80% danych)
        sample_size = int(X.shape[0] * 0.8)
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]

        # Trenowanie nowego modelu na podzbiorze
        svd_sample = TruncatedSVD(n_components=svd.n_components, random_state=42)
        svd_sample.fit(X_sample)

        # Porównanie podobieństwa tematów
        similarity = cosine_similarity(svd.components_, svd_sample.components_)
        stability_scores.append(np.mean(np.max(similarity, axis=1)))

    return np.mean(stability_scores)


def generate_topic_name(words, weights, n_words=4):
    sorted_pairs = sorted(zip(words, weights), key=lambda x: x[1], reverse=True)
    top_words = [pair[0] for pair in sorted_pairs[:n_words]]
    return " + ".join(top_words)


def visualize_topics_lsa(svd, X, save_path):
    print("Performing document transformation...")
    doc_topics = svd.transform(X)

    print("Running t-SNE dimensionality reduction...")
    n_jobs = max(1, multiprocessing.cpu_count() - 2)
    tsne = TSNE(n_components=2,
                random_state=42,
                n_jobs=n_jobs,
                verbose=1)
    tsne_output = tsne.fit_transform(doc_topics)

    print("Finding dominant topics...")
    dominant_topics = np.argmax(doc_topics, axis=1)

    print("Creating visualization...")
    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(tsne_output[:, 0], tsne_output[:, 1],
                          c=dominant_topics,
                          cmap='tab20',
                          alpha=0.6,
                          s=10)

    plt.colorbar(scatter, label='Topic Number')
    plt.title('LSA Topic Distribution (t-SNE visualization)', pad=20)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')

    plt.figtext(0.02, 0.02, f'Total topics: {svd.n_components}',
                fontsize=8, alpha=0.7)

    print(f"Saving visualization to {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def find_optimal_topics(texts, min_topics=5, max_topics=50, step=5):
    print("Creating TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=12000,
        min_df=10,
        max_df=0.75,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words=get_custom_stop_words()
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

    print("\nFinding optimal number of topics...")
    for n_topics in tqdm(range(min_topics, max_topics + 1, step)):
        svd = TruncatedSVD(n_components=n_topics, random_state=42)
        svd.fit(X)

        # Podstawowe metryki
        sentence_topic_dist = svd.transform(X)
        sentence_topics = np.argmax(sentence_topic_dist, axis=1)
        topic_sizes = np.bincount(sentence_topics)

        # Sprawdzenie rozmiaru tematów
        valid_topics = np.sum((topic_sizes >= min_cluster_size) & (topic_sizes <= max_cluster_size))
        if valid_topics < n_topics * 0.8:
            print(f"Warning: Only {valid_topics}/{n_topics} topics meet size criteria")
            continue

        # Obliczenie wszystkich metryk
        coherence = calculate_coherence(svd, vectorizer, X)
        diversity = calculate_diversity(svd, vectorizer)
        stability = calculate_topic_stability(svd, X)
        explained_variance = svd.explained_variance_ratio_.sum()

        # Normalizacja i ważenie metryk
        normalized_coherence = coherence / (1 + coherence)  # Normalizacja do [0,1]
        normalized_diversity = diversity  # Już jest w zakresie [0,1]
        normalized_stability = stability  # Już jest w zakresie [0,1]
        normalized_variance = explained_variance / (1 + explained_variance)

        # Wagi dla różnych metryk
        weights = {
            'coherence': 0.4,
            'diversity': 0.2,
            'stability': 0.2,
            'variance': 0.2
        }

        # Obliczenie końcowego wyniku
        combined_score = (
                weights['coherence'] * normalized_coherence +
                weights['diversity'] * normalized_diversity +
                weights['stability'] * normalized_stability +
                weights['variance'] * normalized_variance
        )

        scores.append((n_topics, coherence, explained_variance, valid_topics, combined_score))

        # Zapisanie najlepszej konfiguracji
        if combined_score > best_score:
            best_score = combined_score
            best_config = {
                'n_topics': n_topics,
                'coherence': coherence,
                'diversity': diversity,
                'stability': stability,
                'variance': explained_variance,
                'score': combined_score
            }

        print(
            f"Topics: {n_topics}, Valid: {valid_topics}, "
            f"Coherence: {coherence:.4f}, Variance: {explained_variance:.4f}, "
            f"Diversity: {diversity:.4f}, Stability: {stability:.4f}, "
            f"Score: {combined_score:.4f}"
        )
        gc.collect()

    if best_config:
        print("\nBest configuration found:")
        for key, value in best_config.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    return scores, vectorizer, X


def save_results(svd, vectorizer, explained_variance_ratio, log_file_path, X):
    feature_names = vectorizer.get_feature_names_out()
    results = f"\nNumber of topics: {svd.n_components}\n"
    results += f"Total explained variance: {explained_variance_ratio.sum():.4f}\n\n"

    topic_order = np.argsort(-explained_variance_ratio)

    for idx, topic_idx in enumerate(topic_order):
        topic = svd.components_[topic_idx]
        top_words_idx = topic.argsort()[:-20:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        word_weights = [topic[i] for i in top_words_idx]

        max_weight = max(abs(w) for w in word_weights)
        norm_weights = [w / max_weight for w in word_weights]

        topic_name = generate_topic_name(top_words, norm_weights)

        variance = explained_variance_ratio[topic_idx]
        results += f"Topic {idx + 1}: {topic_name} (Variance: {variance:.4f})\n"

        for word, weight in zip(top_words, norm_weights):
            results += f"  - {word}: {weight:.4f}\n"
        results += "\n"

    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(results)

    viz_path = log_file_path.replace('.txt', '_visualization.png')
    print("\nGenerating topic visualization...")
    visualize_topics_lsa(svd, X, viz_path)
    print(f"Visualization saved to: {viz_path}")


def main():
    data_path = '../../data/dataset_combined.csv'
    results_path = '../../logs/LSA_results.txt'
    min_topics, max_topics, step = 75, 150, 25

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    download_nltk_resources()

    print("Loading data...")
    data = pd.read_csv(data_path, low_memory=False)

    print("\nPreprocessing texts...")
    processed_sentences = preprocess_data_parallel(data['review'])

    t0 = time()
    scores, vectorizer, X = find_optimal_topics(
        processed_sentences,
        min_topics=min_topics,
        max_topics=max_topics,
        step=step
    )

    if not scores:
        print("No valid topic configurations found!")
        return

    optimal_topics = max(scores, key=lambda x: x[4])[0]  # Używamy combined_score (indeks 4)
    print(f"\nOptimal number of topics: {optimal_topics}")

    final_svd = TruncatedSVD(n_components=optimal_topics, random_state=42)
    final_svd.fit(X)

    save_results(
        final_svd,
        vectorizer,
        final_svd.explained_variance_ratio_,
        results_path,
        X
    )

    print(f"\nAnalysis completed in {time() - t0:.2f} seconds")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()