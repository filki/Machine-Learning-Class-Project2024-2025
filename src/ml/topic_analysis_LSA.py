import os
import pandas as pd
import numpy as np

# Global configuration
N_COMPONENTS = 15  # Number of topics for LSA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import nltk
import re
from time import time
import gc
import warnings

warnings.filterwarnings('ignore')


def download_nltk_resources():
    resources = ['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")


def get_custom_stop_words():
    """Get only standard English stop words without gaming-specific terms."""
    return set(stopwords.words('english'))


def clean_text(text):
    """Enhanced text cleaning with improved pattern handling."""
    if not isinstance(text, str):
        return ''

    # Compile regex patterns
    url_pattern = re.compile(r'http\S+|www\S+|https\S+')
    email_pattern = re.compile(r'\S+@\S+')
    special_char_pattern = re.compile(r'[^\w\s\']')
    number_with_k_pattern = re.compile(r'\d+k\b')
    number_pattern = re.compile(r'\b\d+\b')

    # Clean text
    text = text.lower()
    text = url_pattern.sub('', text)
    text = email_pattern.sub('', text)

    # Replace patterns with tokens
    text = number_with_k_pattern.sub('numberk', text)
    text = number_pattern.sub('number', text)

    # Remove special characters but preserve apostrophes
    text = special_char_pattern.sub(' ', text)

    # Handle contractions
    text = text.replace("'s", " is")
    text = text.replace("'m", " am")
    text = text.replace("'re", " are")
    text = text.replace("'ve", " have")
    text = text.replace("'ll", " will")
    text = text.replace("'d", " would")
    text = text.replace("n't", " not")

    # Remove extra whitespace and ensure words are at least 2 characters
    words = text.split()
    words = [word for word in words if len(word) >= 2]
    return ' '.join(words)


def extract_common_bigrams(texts, min_freq=40):
    """Extract common bigrams with optimized parameters."""
    words = [word_tokenize(text.lower()) for text in texts]
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_documents(words)

    # Filtrowanie bigramów
    finder.apply_freq_filter(min_freq)
    finder.apply_word_filter(lambda w: len(w) < 3)  # Usuwanie krótkich słów z bigramów

    # Wybór bigramów z najwyższym PMI (Pointwise Mutual Information)
    bigrams = finder.nbest(bigram_measures.pmi, 400)

    # Dodatkowe filtrowanie bigramów zawierających stopwords
    stop_words = get_custom_stop_words()
    filtered_bigrams = [
        bigram for bigram in bigrams
        if not (bigram[0] in stop_words or bigram[1] in stop_words)
    ]
    return {' '.join(bigram) for bigram in bigrams}


def preprocess_text(text, lemmatizer=None, stop_words=None, common_bigrams=None):
    if not isinstance(text, str):
        return ''

    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    if stop_words is None:
        stop_words = get_custom_stop_words()

    tokens = word_tokenize(text)
    processed_tokens = []

    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1:
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            if common_bigrams and bigram in common_bigrams:
                processed_tokens.append(bigram)
                i += 2
                continue

        token = tokens[i]
        if token not in stop_words and len(token) >= 2:
            lemmatized = lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized)
        i += 1

    return ' '.join(processed_tokens)


def perform_lsa_analysis(data, n_components=N_COMPONENTS, batch_size=4096):
    """Perform LSA analysis with optimized parameters for better topic separation."""
    print("Preparing data for vectorization...")

    print("Extracting common bigrams...")
    common_bigrams = extract_common_bigrams(data['cleaned_review'])
    print(f"Found {len(common_bigrams)} common bigrams")

    print("Preprocessing text...")
    lemmatizer = WordNetLemmatizer()
    stop_words = get_custom_stop_words()
    data['processed_review'] = data['cleaned_review'].apply(
        lambda x: preprocess_text(x, lemmatizer, stop_words, common_bigrams)
    )

    data = data[data['processed_review'].str.strip().str.len() > 0]
    print(f"Number of valid documents: {len(data)}")

    print("Creating TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=12000,  # Zmniejszona liczba cech
        min_df=15,  # Lekko zwiększony próg minimalnej częstości
        max_df=0.80,  # Bardziej restrykcyjne filtrowanie częstych terminów
        token_pattern=r'(?u)\b\w+\b',  # Prostszy wzorzec tokenizacji
        ngram_range=(1, 2),
        lowercase=True,  # Ujednolicenie wielkości liter
        sublinear_tf=True,
        norm='l2',  # Normalizacja L2 dla lepszego rozkładu wag
        use_idf=True,  # Użycie IDF
        smooth_idf=True  # Wygładzanie IDF
    )

    X = vectorizer.fit_transform(data['processed_review'])
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    print("Performing LSA...")
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42,
        n_iter=15
    )

    lsa_result = svd.fit_transform(X)
    explained_variance_ratio = svd.explained_variance_ratio_
    total_variance = explained_variance_ratio.sum()
    print(f"Total explained variance: {total_variance:.4f}")

    return svd, vectorizer, X, explained_variance_ratio, common_bigrams


def save_results(svd, vectorizer, explained_variance_ratio, log_file_path, common_bigrams):
    feature_names = vectorizer.get_feature_names_out()
    results = f"\n{'=' * 30}\nNumber of topics: {svd.n_components}\n"
    results += f"Total explained variance: {explained_variance_ratio.sum():.4f}\n\n"

    # Sortowanie tematów według wyjaśnionej wariancji
    topic_order = np.argsort(-explained_variance_ratio)

    for idx, topic_idx in enumerate(topic_order):
        topic = svd.components_[topic_idx]

        # Znajdź najważniejsze słowa i ich wagi
        top_words_idx = topic.argsort()[:-25 - 1:-1]  # Top 25 słów
        top_words = [feature_names[i] for i in top_words_idx]
        word_weights = [topic[i] for i in top_words_idx]

        # Normalizacja wag dla lepszej interpretacji
        max_weight = max(abs(w) for w in word_weights)
        norm_weights = [w / max_weight for w in word_weights]

        variance = explained_variance_ratio[topic_idx]
        results += f"Topic {idx + 1} (Variance explained: {variance:.4f}):\n"

        # Grupowanie na unigramy i bigramy
        unigrams = [(w, v) for w, v in zip(top_words, norm_weights) if ' ' not in w]
        bigrams = [(w, v) for w, v in zip(top_words, norm_weights) if ' ' in w]

        if unigrams:
            results += "  Unigrams:\n"
            for word, weight in unigrams:
                results += f"    - {word}: {weight:.4f}\n"

        if bigrams:
            results += "  Bigrams:\n"
            for word, weight in bigrams:
                results += f"    - {word}: {weight:.4f}\n"

        results += "\n"

    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(results)


def main():
    raw_data_path = '../../data/dataset_combined.csv'
    cleaned_data_path = '../../data/cleaned_dataset_lsa.csv'
    log_file_path = '../../logs/LSA_results.txt'

    os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    download_nltk_resources()

    print("Loading data...")
    data = pd.read_csv(raw_data_path)

    print("Cleaning text...")
    data['cleaned_review'] = data['review'].apply(clean_text)

    print(f"Dataset size after cleaning: {len(data)} reviews")

    t0 = time()
    svd, vectorizer, X, explained_variance_ratio, common_bigrams = perform_lsa_analysis(
        data
    )

    save_results(svd, vectorizer, explained_variance_ratio, log_file_path, common_bigrams)

    print(f"Analysis completed in {time() - t0:.2f} seconds")
    print(f"Total variance explained: {explained_variance_ratio.sum():.4f}")


if __name__ == "__main__":
    main()