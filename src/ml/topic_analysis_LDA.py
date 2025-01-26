import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
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


def get_optimal_workers():
    """Określa optymalną liczbę wątków do przetwarzania równoległego"""
    return max(1, multiprocessing.cpu_count() - 1)


def download_nltk_resources():
    """Pobiera wymagane zasoby NLTK"""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass


def get_custom_stop_words():
    """Zwraca rozszerzoną listę stop words"""
    domain_stop_words = {
        'game', 'games', 'play', 'played', 'playing', 'gameplay',
        'steam', 'review', 'reviews', 'recommend', 'recommended',
        'hour', 'hours', 'like', 'good', 'bad', 'great', 'best',
        'worst', 'really', 'much', 'many', 'lot', 'well', 'make',
        'made', 'time', 'player', 'players'
    }
    return set(stopwords.words('english')).union(domain_stop_words)


def clean_text(text):
    """Czyści tekst z niepożądanych elementów"""
    if not isinstance(text, str):
        return ''

    patterns = {
        'url': re.compile(r'http\S+|www\S+|https\S+'),
        'special_chars': re.compile(r'[^\w\s\']'),
        'numbers': re.compile(r'\d+'),
        'extra_spaces': re.compile(r'\s+')
    }

    text = patterns['url'].sub('', text)
    text = patterns['special_chars'].sub(' ', text)
    text = patterns['numbers'].sub('', text)
    text = patterns['extra_spaces'].sub(' ', text)

    return text.lower().strip()


def process_text(text, lemmatizer, stop_words):
    """Przetwarza tekst z wykorzystaniem lemmatyzacji i usuwaniem stop words"""
    if not isinstance(text, str):
        return ''

    sentences = []
    for sent in sent_tokenize(text):
        tokens = word_tokenize(clean_text(sent))
        tokens = [lemmatizer.lemmatize(word) for word in tokens
                  if word not in stop_words and len(word) > 2]
        if tokens:
            sentences.append(' '.join(tokens))

    return ' '.join(sentences)


def process_chunk(chunk, lemmatizer, stop_words):
    """Pomocnicza funkcja do przetwarzania pojedynczego chunka"""
    return [process_text(text, lemmatizer, stop_words) for text in chunk]


def parallel_process_texts(texts, chunk_size=1000):
    """Przetwarza teksty równolegle w chunk'ach"""
    lemmatizer = WordNetLemmatizer()
    stop_words = get_custom_stop_words()

    processed_texts = []
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    process_chunk_with_params = partial(process_chunk,
                                        lemmatizer=lemmatizer,
                                        stop_words=stop_words)

    with ProcessPoolExecutor(max_workers=get_optimal_workers()) as executor:
        with tqdm(total=len(texts), desc="Processing texts") as pbar:
            for chunk_result in executor.map(process_chunk_with_params, chunks):
                processed_texts.extend(chunk_result)
                pbar.update(len(chunk_result))
                gc.collect()

    return processed_texts


def calculate_topic_coherence(lda, vectorizer, X, top_n=20):
    """Oblicza spójność tematów na podstawie PMI"""
    feature_names = vectorizer.get_feature_names_out()
    coherence_scores = []
    X_dense = X.toarray()

    for topic in lda.components_:
        top_term_indices = topic.argsort()[:-top_n - 1:-1]
        top_terms = [feature_names[i] for i in top_term_indices]

        pair_scores = []
        for i in range(len(top_terms)):
            for j in range(i + 1, len(top_terms)):
                term1_docs = X_dense[:, vectorizer.vocabulary_[top_terms[i]]] > 0
                term2_docs = X_dense[:, vectorizer.vocabulary_[top_terms[j]]] > 0

                cooccurrence = np.sum(term1_docs & term2_docs) + 1
                term1_count = np.sum(term1_docs) + 1
                term2_count = np.sum(term2_docs) + 1
                total_docs = len(X_dense) + 1

                pmi = np.log((cooccurrence * total_docs) / (term1_count * term2_count))
                pair_scores.append(pmi)

        coherence_scores.append(np.mean(pair_scores))

    return np.mean(coherence_scores)


def calculate_topic_diversity(lda, vectorizer, top_n=20):
    """Oblicza różnorodność tematów"""
    feature_names = vectorizer.get_feature_names_out()
    topic_words = []

    for topic in lda.components_:
        top_indices = topic.argsort()[:-top_n - 1:-1]
        topic_words.append(set(feature_names[i] for i in top_indices))

    diversity_scores = []
    for i in range(len(topic_words)):
        for j in range(i + 1, len(topic_words)):
            jaccard = len(topic_words[i].intersection(topic_words[j])) / len(topic_words[i].union(topic_words[j]))
            diversity_scores.append(1 - jaccard)

    return np.mean(diversity_scores)


def find_optimal_topics(X, vectorizer, min_topics=25, max_topics=1000, step=25):
    """Znajduje optymalną liczbę topików na podstawie metryk"""
    print("\nSearching for optimal number of topics...")
    results = []

    for n_topics in tqdm(range(min_topics, max_topics + 1, step), desc="Evaluating topics"):
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method='online',
            batch_size=4096,
            max_iter=20,
            n_jobs=get_optimal_workers(),
            random_state=42
        )

        # Trenowanie modelu
        doc_topics = lda.fit_transform(X)

        # Obliczanie metryk
        coherence = calculate_topic_coherence(lda, vectorizer, X)
        diversity = calculate_topic_diversity(lda, vectorizer)
        perplexity = lda.perplexity(X)

        # Normalizacja metryk
        norm_coherence = coherence / (1 + abs(coherence))
        norm_diversity = diversity
        norm_perplexity = 1 / (1 + np.log1p(perplexity))

        # Ważona suma metryk
        combined_score = (0.4 * norm_coherence +
                          0.4 * norm_diversity +
                          0.2 * norm_perplexity)

        results.append({
            'n_topics': n_topics,
            'coherence': coherence,
            'diversity': diversity,
            'perplexity': perplexity,
            'combined_score': combined_score
        })

        gc.collect()

    # Wybór najlepszej liczby topików
    best_result = max(results, key=lambda x: x['combined_score'])

    print(f"\nOptimal number of topics found: {best_result['n_topics']}")
    print(f"Metrics for optimal solution:")
    print(f"Coherence: {best_result['coherence']:.4f}")
    print(f"Diversity: {best_result['diversity']:.4f}")
    print(f"Perplexity: {best_result['perplexity']:.4f}")
    print(f"Combined Score: {best_result['combined_score']:.4f}")

    return best_result['n_topics'], results


def analyze_and_visualize(data_path, n_topics=None, min_topics=25, max_topics=1000, step=25, results_path=None):
    """Główna funkcja przeprowadzająca analizę LDA i generująca wizualizację"""
    t0 = time()
    download_nltk_resources()

    # Wczytanie i przetworzenie danych
    print("Loading and preprocessing data...")
    data = pd.read_csv(data_path, low_memory=False)
    processed_texts = parallel_process_texts(data['review'].tolist())

    # Wektoryzacja
    print("Vectorizing texts...")
    vectorizer = CountVectorizer(
        max_features=20000,
        min_df=20,
        max_df=0.85,
        token_pattern=r'\b\w+\b'
    )
    X = vectorizer.fit_transform(processed_texts)

    # Znalezienie optymalnej liczby topików lub użycie podanej wartości
    if n_topics is None:
        print("Finding optimal number of topics...")
        n_topics, topic_search_results = find_optimal_topics(
            X, vectorizer, min_topics, max_topics, step
        )

        # Zapisywanie wyników wyszukiwania
        results_df = pd.DataFrame(topic_search_results)
        results_df.to_csv(results_path.replace('.txt', '_search_results.csv'), index=False)

        # Wizualizacja wyników wyszukiwania
        plt.figure(figsize=(12, 8))
        plt.plot(results_df['n_topics'], results_df['combined_score'], marker='o')
        plt.xlabel('Number of Topics')
        plt.ylabel('Combined Score')
        plt.title('Topic Number Optimization')
        plt.grid(True)
        plt.savefig(results_path.replace('.txt', '_optimization.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Trenowanie finalnego modelu LDA
    print(f"Training final LDA model with {n_topics} topics...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method='online',
        batch_size=4096,
        max_iter=20,
        n_jobs=get_optimal_workers(),
        random_state=42
    )

    doc_topics = lda.fit_transform(X)

    # Obliczanie metryk
    print("Calculating metrics...")
    metrics = {
        'coherence': calculate_topic_coherence(lda, vectorizer, X),
        'diversity': calculate_topic_diversity(lda, vectorizer),
        'perplexity': lda.perplexity(X)
    }

    # Zapisywanie wyników
    results = {
        'metrics': metrics,
        'top_words': get_top_words_per_topic(lda, vectorizer, n_words=20)
    }

    with open(results_path, 'w') as f:
        f.write("LDA Analysis Results\n\n")
        f.write(f"Number of topics: {n_topics}\n\n")
        f.write("Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\nTop words per topic:\n")
        for topic_idx, words in results['top_words'].items():
            f.write(f"\nTopic {topic_idx + 1}:\n")
            f.write(", ".join(words))

    # Wizualizacja t-SNE
    print("Generating visualization...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=get_optimal_workers())
    tsne_output = tsne.fit_transform(doc_topics)

    plt.figure(figsize=(12, 8))
    dominant_topics = doc_topics.argmax(axis=1)
    scatter = plt.scatter(tsne_output[:, 0], tsne_output[:, 1],
                          c=dominant_topics, cmap='tab20', alpha=0.6, s=10)
    plt.colorbar(scatter)
    plt.title('Topic Distribution (t-SNE)')
    plt.savefig(results_path.replace('.txt', '_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Analysis completed in {time() - t0:.2f} seconds")
    print(f"Results saved to: {results_path}")
    return lda, vectorizer, results


def get_top_words_per_topic(lda, vectorizer, n_words=20):
    """Zwraca najważniejsze słowa dla każdego tematu"""
    feature_names = vectorizer.get_feature_names_out()
    top_words = {}

    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-n_words - 1:-1]
        top_words[topic_idx] = [feature_names[i] for i in top_indices]

    return top_words


if __name__ == "__main__":
    data_path = '../../data/dataset_combined.csv'
    results_path = '../../logs/LDA_results.txt'
    min_topics, max_topics, step = 50, 300, 25

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    analyze_and_visualize(
        data_path=data_path,
        n_topics=None,
        min_topics=min_topics,
        max_topics=max_topics,
        step=step,
        results_path=results_path
    )