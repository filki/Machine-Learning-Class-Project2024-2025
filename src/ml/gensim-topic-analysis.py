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
from sklearn.manifold import TSNE
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
import logging
from gensim import corpora, models
from gensim.models import LsiModel, LdaModel, CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

# Konfiguracja loggingu
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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

    processed_texts = []
    for text in chunk:
        if not isinstance(text, str):
            continue
        # Tokenizacja na słowa zamiast zdań
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|[^\w\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens
                 if word not in stop_words and len(word) > 2]
        if tokens:
            processed_texts.append(tokens)
    return processed_texts

def preprocess_data_parallel(data, n_cores=None):
    if n_cores is None:
        n_cores = max(1, multiprocessing.cpu_count() - 2)

    chunk_size = max(1, len(data) // (n_cores * 4))
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    print(f"Using {n_cores} cores for processing {len(chunks)} chunks")
    print(f"Chunk size: {chunk_size} documents")

    lemmatizer = WordNetLemmatizer()
    stop_words = get_custom_stop_words()

    processed_texts = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        process_func = partial(process_text_chunk,
                             lemmatizer=lemmatizer,
                             stop_words=stop_words)
        results = list(tqdm(executor.map(process_func, chunks),
                          total=len(chunks),
                          desc="Processing chunks"))

    for chunk_texts in results:
        processed_texts.extend(chunk_texts)

    print(f"Total documents processed: {len(processed_texts)}")
    return processed_texts

def calculate_topic_diversity(model, top_n=10):
    if isinstance(model, LsiModel):
        topics = model.show_topics(num_topics=-1, num_words=top_n, formatted=False)
    else:  # LdaModel
        topics = model.show_topics(num_topics=-1, num_words=top_n, formatted=False)
    
    all_words = set()
    for topic in topics:
        words = [word for word, _ in topic[1]]
        all_words.update(words)
    
    diversity = len(all_words) / (model.num_topics * top_n)
    return diversity

def calculate_topic_stability(model, corpus, dictionary, n_samples=5):
    stability_scores = []
    
    for _ in range(n_samples):
        # Losowy podzbiór dokumentów (80% danych)
        sample_indices = np.random.choice(len(corpus), size=int(len(corpus) * 0.8), replace=False)
        corpus_sample = [corpus[i] for i in sample_indices]
        
        # Trenowanie nowego modelu na podzbiorze
        if isinstance(model, LsiModel):
            model_sample = LsiModel(corpus=corpus_sample, 
                                  id2word=dictionary,
                                  num_topics=model.num_topics)
        else:  # LdaModel
            model_sample = LdaModel(corpus=corpus_sample,
                                  id2word=dictionary,
                                  num_topics=model.num_topics,
                                  alpha='auto',
                                  passes=20)
        
        # Porównanie podobieństwa tematów
        if isinstance(model, LsiModel):
            similarity = np.abs(np.corrcoef(model.projection.projection,
                                          model_sample.projection.projection))
        else:  # LdaModel
            topic_terms1 = model.get_topics()
            topic_terms2 = model_sample.get_topics()
            similarity = np.abs(np.corrcoef(topic_terms1, topic_terms2))
        
        stability_scores.append(np.mean(np.max(similarity, axis=1)))
    
    return np.mean(stability_scores)

def visualize_topics(model, corpus, method='tsne', save_path=None):
    # Przekształcenie dokumentów na rozkład tematów
    doc_topics = np.array([model[doc] for doc in corpus])
    doc_topics = np.array([[weight for _, weight in doc] for doc in doc_topics])
    
    # Uzupełnienie brakujących tematów zerami
    padded_doc_topics = np.zeros((len(doc_topics), model.num_topics))
    for i, doc in enumerate(doc_topics):
        for j, weight in enumerate(doc):
            padded_doc_topics[i, j] = weight
    
    # Redukcja wymiarowości za pomocą t-SNE
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    tsne_output = tsne.fit_transform(padded_doc_topics)
    
    # Znalezienie dominującego tematu dla każdego dokumentu
    dominant_topics = np.argmax(padded_doc_topics, axis=1)
    
    # Wizualizacja
    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(tsne_output[:, 0], tsne_output[:, 1],
                         c=dominant_topics,
                         cmap='tab20',
                         alpha=0.6,
                         s=10)
    
    plt.colorbar(scatter, label='Topic Number')
    plt.title(f'Topic Distribution ({method.upper()} visualization)', pad=20)
    plt.xlabel(f'{method.upper()} dimension 1')
    plt.ylabel(f'{method.upper()} dimension 2')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def find_optimal_topics(texts, dictionary, corpus, model_type='lsa', 
                       min_topics=5, max_topics=50, step=5):
    scores = []
    best_score = float('-inf')
    best_config = None
    
    print(f"\nFinding optimal number of topics for {model_type.upper()}...")
    for n_topics in tqdm(range(min_topics, max_topics + 1, step)):
        # Trenowanie modelu
        if model_type == 'lsa':
            model = LsiModel(corpus=corpus, id2word=dictionary, num_topics=n_topics)
        else:  # lda
            model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics,
                           alpha='auto', passes=20)
        
        # Obliczenie metryk
        coherence_model = CoherenceModel(model=model, texts=texts,
                                       dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        diversity = calculate_topic_diversity(model)
        stability = calculate_topic_stability(model, corpus, dictionary)
        
        # Obliczenie combined score
        normalized_coherence = coherence / (1 + coherence)
        normalized_diversity = diversity
        normalized_stability = stability
        
        weights = {
            'coherence': 0.4,
            'diversity': 0.3,
            'stability': 0.3
        }
        
        combined_score = (
            weights['coherence'] * normalized_coherence +
            weights['diversity'] * normalized_diversity +
            weights['stability'] * normalized_stability
        )
        
        scores.append((n_topics, coherence, diversity, stability, combined_score))
        
        if combined_score > best_score:
            best_score = combined_score
            best_config = {
                'n_topics': n_topics,
                'coherence': coherence,
                'diversity': diversity,
                'stability': stability,
                'score': combined_score
            }
        
        print(
            f"Topics: {n_topics}, "
            f"Coherence: {coherence:.4f}, "
            f"Diversity: {diversity:.4f}, "
            f"Stability: {stability:.4f}, "
            f"Score: {combined_score:.4f}"
        )
        gc.collect()
    
    if best_config:
        print("\nBest configuration found:")
        for key, value in best_config.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    return scores, best_config

def save_model_results(model, corpus, dictionary, model_type, log_file_path):
    results = f"\nNumber of topics: {model.num_topics}\n\n"
    
    # Pobranie i sortowanie tematów według ważności
    topics = model.show_topics(num_topics=-1, num_words=20, formatted=False)
    
    if isinstance(model, LsiModel):
        # Dla LSA sortujemy według singular values
        topic_strengths = np.abs(model.projection.s)
        topic_order = np.argsort(-topic_strengths)
    else:  # LdaModel
        # Dla LDA używamy częstości tematów w korpusie
        topic_counts = np.zeros(model.num_topics)
        for doc in corpus:
            doc_topics = model[doc]
            for topic_id, weight in doc_topics:
                topic_counts[topic_id] += weight
        topic_order = np.argsort(-topic_counts)
    
    # Zapisanie tematów
    for idx, topic_idx in enumerate(topic_order):
        topic_words = topics[topic_idx][1]
        topic_str = " + ".join([f"{word} ({weight:.3f})" 
                              for word, weight in topic_words[:10]])
        
        if isinstance(model, LsiModel):
            strength = topic_strengths[topic_idx]
            results += f"Topic {idx + 1}: (Strength: {strength:.4f})\n"
        else:  # LdaModel
            frequency = topic_counts[topic_idx] / sum(topic_counts)
            results += f"Topic {idx + 1}: (Frequency: {frequency:.4f})\n"
            
        results += f"{topic_str}\n\n"
    
    # Zapis do pliku
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(results)
    
    # Wizualizacja
    viz_path = log_file_path.replace('.txt', '_visualization.png')
    print(f"\nGenerating topic visualization...")
    visualize_topics(model, corpus, save_path=viz_path)
    print(f"Visualization saved to: {viz_path}")
    
    # Dodatkowa wizualizacja dla LDA
    if isinstance(model, LdaModel):
        pyldavis_path = log_file_path.replace('.txt', '_pyLDAvis.html')
        print(f"Generating pyLDAvis visualization...")
        vis_data = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
        pyLDAvis.save_html(vis_data, pyldavis_path)
        print(f"pyLDAvis visualization saved to: {pyldavis_path}")

def perform_nmf_analysis(texts, n_topics):
    """
    Przeprowadza analizę NMF dla danej liczby tematów
    """
    # Przygotowanie danych
    vectorizer = TfidfVectorizer(
        max_features=12000,
        min_df=10,
        max_df=0.75,
        stop_words=get_custom_stop_words()
    )
    X = vectorizer.fit_transform(texts)
    
    # Trenowanie modelu NMF
    model = NMF(
        n_components=n_topics,
        random_state=42,
        init='nndsvd'  # Deterministyczna inicjalizacja
    )
    W = model.fit_transform(X)  # dokumenty-tematy
    H = model.components_  # tematy-słowa
    
    # Normalizacja macierzy
    H = normalize(H, norm='l1', axis=1)
    
    # Obliczenie metryk
    reconstruction_error = model.reconstruction_err_
    
    # Obliczenie spójności tematów (podobnie jak w LSA)
    topic_coherence = np.mean([
        np.mean(cosine_similarity(X[:, topic_terms].toarray().T))
        for topic_terms in (H > H.mean(axis=1)[:, np.newaxis]).astype(bool)
    ])
    
    # Obliczenie różnorodności
    feature_names = vectorizer.get_feature_names_out()
    top_n = 10
    all_top_words = set()
    for topic_idx in range(H.shape[0]):
        top_words = [feature_names[i] 
                    for i in H[topic_idx].argsort()[:-top_n-1:-1]]
        all_top_words.update(top_words)
    diversity = len(all_top_words) / (n_topics * top_n)
    
    # Obliczenie stabilności
    stability_scores = []
    n_samples = 5
    sample_size = int(X.shape[0] * 0.8)
    
    for _ in range(n_samples):
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]
        model_sample = NMF(n_components=n_topics, random_state=42, init='nndsvd')
        H_sample = model_sample.fit_transform(X_sample).components_
        H_sample = normalize(H_sample, norm='l1', axis=1)
        
        # Porównanie podobieństwa tematów
        similarity = cosine_similarity(H, H_sample)
        stability_scores.append(np.mean(np.max(similarity, axis=1)))
    
    stability = np.mean(stability_scores)
    
    return {
        'reconstruction_error': reconstruction_error,
        'coherence': topic_coherence,
        'diversity': diversity,
        'stability': stability,
        'model': model,
        'vectorizer': vectorizer,
        'W': W,
        'H': H
    }

def save_nmf_results(results, feature_names, log_file_path):
    """
    Zapisuje wyniki analizy NMF do pliku
    """
    H = results['H']
    n_topics = H.shape[0]
    
    output = f"Number of topics: {n_topics}\n"
    output += f"Reconstruction error: {results['reconstruction_error']:.4f}\n"
    output += f"Coherence: {results['coherence']:.4f}\n"
    output += f"Diversity: {results['diversity']:.4f}\n"
    output += f"Stability: {results['stability']:.4f}\n\n"
    
    # Sortowanie tematów według "ważności" (sumy wag słów)
    topic_importance = np.sum(H, axis=1)
    topic_order = np.argsort(-topic_importance)
    
    for idx, topic_idx in enumerate(topic_order):
        top_words_idx = np.argsort(-H[topic_idx])[:20]
        top_words = [(feature_names[i], H[topic_idx][i]) 
                    for i in top_words_idx]
        
        output += f"Topic {idx+1} (Importance: {topic_importance[topic_idx]:.4f}):\n"
        for word, weight in top_words:
            output += f"  - {word}: {weight:.4f}\n"
        output += "\n"
    
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(output)
    
    # Wizualizacja
    viz_path = log_file_path.replace('.txt', '_visualization.png')
    print("\nGenerating topic visualization...")
    
    # t-SNE na macierzy dokumenty-tematy
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    tsne_output = tsne.fit_transform(results['W'])
    
    plt.figure(figsize=(16, 10))
    dominant_topics = np.argmax(results['W'], axis=1)
    scatter = plt.scatter(tsne_output[:, 0], tsne_output[:, 1],
                         c=dominant_topics,
                         cmap='tab20',
                         alpha=0.6,
                         s=10)
    
    plt.colorbar(scatter, label='Topic Number')
    plt.title('NMF Topic Distribution (t-SNE visualization)', pad=20)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {viz_path}")

def main():
    data_path = '../../data/dataset_combined.csv'
    results_path_lsa = '../../logs/gensim_LSA_results.txt'
    results_path_lda = '../../logs/gensim_LDA_results.txt'
    results_path_nmf = '../../logs/gensim_NMF_results.txt'
    min_topics, max_topics, step = 50, 300, 25
    
    # Tworzenie katalogów dla wyników
    os.makedirs(os.path.dirname(results_path_lsa), exist_ok=True)
    download_nltk_resources()
    
    print("Loading data...")
    data = pd.read_csv(data_path, low_memory=False)
    
    print("\nPreprocessing texts...")
    processed_texts = preprocess_data_parallel(data['review'])
    
    # Tworzenie słownika i korpusu
    print("\nCreating dictionary and corpus...")
    dictionary = corpora.Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=10, no_above=0.75)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    # Analiza LSA
    print("\nPerforming LSA analysis...")
    t0 = time()
    lsa_scores, lsa_best = find_optimal_topics(
        processed_texts, dictionary, corpus,
        model_type='lsa',
        min_topics=min_topics,
        max_topics=max_topics,
        step=step
    )
    
    if lsa_best:
        final_lsa = LsiModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=lsa_best['n_topics']
        )
        save_model_results(
            final_lsa,
            corpus,
            dictionary,
            'lsa',
            results_path_lsa
        )
        print(f"\nLSA analysis completed in {time() - t0:.2f} seconds")
        
    # Analiza LDA
    print("\nPerforming LDA analysis...")
    t0 = time()
    lda_scores, lda_best = find_optimal_topics(
        processed_texts, dictionary, corpus,
        model_type='lda',
        min_topics=min_topics,
        max_topics=max_topics,
        step=step
    )
    
    if lda_best:
        final_lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=lda_best['n_topics'],
            alpha='auto',
            passes=20
        )
        save_model_results(
            final_lda,
            corpus,
            dictionary,
            'lda',
            results_path_lda
        )
        print(f"\nLDA analysis completed in {time() - t0:.2f} seconds")
    
    # Analiza NMF
    print("\nPerforming NMF analysis...")
    t0 = time()
    
    # Dla NMF używamy oryginalnych tekstów (nie tokenów)
    original_texts = [' '.join(text) for text in processed_texts]
    
    best_nmf_score = float('-inf')
    best_nmf_config = None
    
    for n_topics in tqdm(range(min_topics, max_topics + 1, step)):
        print(f"\nTesting NMF with {n_topics} topics...")
        results = perform_nmf_analysis(original_texts, n_topics)
        
        # Normalizacja metryk
        norm_coherence = results['coherence'] / (1 + results['coherence'])
        norm_diversity = results['diversity']
        norm_stability = results['stability']
        norm_error = 1 / (1 + results['reconstruction_error'])
        
        # Obliczenie combined score
        weights = {
            'coherence': 0.3,
            'diversity': 0.2,
            'stability': 0.2,
            'error': 0.3
        }
        
        combined_score = (
            weights['coherence'] * norm_coherence +
            weights['diversity'] * norm_diversity +
            weights['stability'] * norm_stability +
            weights['error'] * norm_error
        )
        
        print(
            f"Coherence: {results['coherence']:.4f}, "
            f"Diversity: {results['diversity']:.4f}, "
            f"Stability: {results['stability']:.4f}, "
            f"Error: {results['reconstruction_error']:.4f}, "
            f"Score: {combined_score:.4f}"
        )
        
        if combined_score > best_nmf_score:
            best_nmf_score = combined_score
            best_nmf_config = {
                'n_topics': n_topics,
                'results': results
            }
    
    if best_nmf_config:
        print(f"\nBest NMF configuration found: {best_nmf_config['n_topics']} topics")
        save_nmf_results(
            best_nmf_config['results'],
            best_nmf_config['results']['vectorizer'].get_feature_names_out(),
            results_path_nmf
        )
        print(f"\nNMF analysis completed in {time() - t0:.2f} seconds")

if __name__ == "__main__":
    main()