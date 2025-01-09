import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
import nltk
import re
from time import time
from joblib import Parallel, delayed
import gc
import warnings

warnings.filterwarnings('ignore')


def download_nltk_resources():
    """Download all required NLTK resources."""
    resources = ['stopwords', 'punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")


def get_custom_stop_words():
    """Get combined set of standard English stop words and gaming-specific stop words."""
    gaming_stop_words = {
        # Podstawowe słowa związane z grami
        'game', 'games', 'play', 'played', 'playing', 'gameplay',
        'steam', 'review', 'reviews', 'recommend', 'recommended',

        # Czasowe odniesienia
        'hour', 'hours', 'time', 'day', 'days', 'week', 'weeks',

        # Ogólne oceny
        'like', 'good', 'bad', 'great', 'best', 'worst', 'really',
        'much', 'many', 'lot', 'well', 'make', 'made', 'better',
        'worse', 'awesome', 'amazing', 'nice', 'perfect', 'terrible',

        # Techniczne terminy
        'fps', 'bug', 'bugs', 'crash', 'crashed', 'install', 'installed',
        'download', 'update', 'updated', 'version', 'release',

        # Platformy i sklepy
        'steam', 'epic', 'gog', 'origin', 'uplay', 'console', 'pc',

        # Wyrażenia związane z zakupem
        'buy', 'bought', 'purchase', 'purchased', 'price', 'cost',
        'worth', 'money', 'free', 'paid',

        # Popularne przyimki i spójniki
        'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'up',
        'about', 'into', 'over', 'after', 'of', 'and', 'or', 'but'
    }
    return set(stopwords.words('english')).union(gaming_stop_words)


def extract_common_bigrams(texts, min_freq=50):
    """Extract common bigrams from the corpus."""
    words = [word_tokenize(text.lower()) for text in texts]
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_documents(words)

    # Apply frequency filter
    finder.apply_freq_filter(min_freq)

    # Get top bigrams using PMI (Pointwise Mutual Information)
    bigrams = finder.nbest(bigram_measures.pmi, 500)
    return {' '.join(bigram) for bigram in bigrams}


def clean_text(text):
    """Enhanced text cleaning with gaming-specific patterns."""
    if not isinstance(text, str):
        return ''

    # Compile regex patterns
    url_pattern = re.compile(r'http\S+|www\S+|https\S+')
    email_pattern = re.compile(r'\S+@\S+')
    special_char_pattern = re.compile(r'[^\w\s\']')
    number_with_k_pattern = re.compile(r'\d+k\b')  # Matches numbers with 'k' suffix
    fps_pattern = re.compile(r'\d+\s*fps')  # Matches FPS numbers
    resolution_pattern = re.compile(r'\d+x\d+')  # Matches screen resolutions

    # Preserve gaming-specific patterns
    text = text.lower()
    text = url_pattern.sub('', text)
    text = email_pattern.sub('', text)

    # Replace gaming-specific patterns with tokens
    text = number_with_k_pattern.sub('numberk', text)
    text = fps_pattern.sub('fpsnumber', text)
    text = resolution_pattern.sub('resolution', text)

    # Remove remaining special characters
    text = special_char_pattern.sub(' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def preprocess_text(text, lemmatizer=None, stop_words=None, common_bigrams=None):
    """Enhanced text preprocessing with bigram support."""
    if not isinstance(text, str):
        return ''

    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    if stop_words is None:
        stop_words = get_custom_stop_words()

    # Tokenize
    tokens = word_tokenize(text)

    # Lemmatize and filter tokens
    processed_tokens = []
    for i in range(len(tokens)):
        token = tokens[i]

        # Check for bigrams
        if i < len(tokens) - 1:
            bigram = f"{token} {tokens[i + 1]}"
            if common_bigrams and bigram in common_bigrams:
                processed_tokens.append(bigram)
                continue

        # Process single tokens
        if token not in stop_words and len(token) > 2:
            lemmatized = lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized)

    return ' '.join(processed_tokens)


def perform_lsa_analysis(data, n_components=15, batch_size=4096):
    """Perform LSA analysis with improved parameters."""
    print("Preparing data for vectorization...")

    # Extract common bigrams first
    print("Extracting common bigrams...")
    common_bigrams = extract_common_bigrams(data['cleaned_review'])
    print(f"Found {len(common_bigrams)} common bigrams")

    # Preprocess with bigrams
    print("Preprocessing text with bigrams...")
    lemmatizer = WordNetLemmatizer()
    stop_words = get_custom_stop_words()
    data['processed_review'] = data['cleaned_review'].apply(
        lambda x: preprocess_text(x, lemmatizer, stop_words, common_bigrams)
    )

    # Remove empty reviews
    data = data[data['processed_review'].str.strip().str.len() > 0]
    print(f"Number of valid documents for analysis: {len(data)}")

    print("Creating TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Zmniejszone z 15000
        min_df=20,  # Zwiększone z 10
        max_df=0.85,  # Zmniejszone z 0.90
        token_pattern=r'(?u)\b[\w\s]+\b',  # Modified to catch bigrams
        ngram_range=(1, 2),  # Include bigrams
        lowercase=False,
        sublinear_tf=True  # Apply sublinear scaling to term frequencies
    )

    print("Vectorizing documents...")
    X = vectorizer.fit_transform(data['processed_review'])
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    print("Performing LSA...")
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42,
        n_iter=10,
        algorithm='randomized'  # Faster for large datasets
    )

    lsa_result = svd.fit_transform(X)
    explained_variance_ratio = svd.explained_variance_ratio_
    total_variance = explained_variance_ratio.sum()
    print(f"Total explained variance: {total_variance:.4f}")

    return svd, vectorizer, X, explained_variance_ratio, common_bigrams


def save_results(svd, vectorizer, explained_variance_ratio, log_file_path, common_bigrams):
    """Enhanced results saving with bigram information."""
    feature_names = vectorizer.get_feature_names_out()
    results = f"\n{'=' * 30}\nNumber of topics: {svd.n_components}\n"
    results += f"Total explained variance: {explained_variance_ratio.sum():.4f}\n\n"

    # Add bigram information
    results += "Common Bigrams Found:\n"
    for bigram in sorted(list(common_bigrams)[:20]):  # Show top 20 bigrams
        results += f"  - {bigram}\n"
    results += "\n"

    results += "Topics:\n"
    for topic_idx, topic in enumerate(svd.components_):
        top_words_idx = topic.argsort()[:-20 - 1:-1]  # Increased from 15 to 20 words
        top_words = [feature_names[i] for i in top_words_idx]
        word_weights = [topic[i] for i in top_words_idx]

        variance = explained_variance_ratio[topic_idx]
        results += f"Topic {topic_idx + 1} (Variance explained: {variance:.4f}):\n"

        # Group unigrams and bigrams
        unigrams = [(w, v) for w, v in zip(top_words, word_weights) if ' ' not in w]
        bigrams = [(w, v) for w, v in zip(top_words, word_weights) if ' ' in w]

        # Print unigrams first
        results += "  Unigrams:\n"
        for word, weight in unigrams:
            results += f"    - {word}: {weight:.4f}\n"

        # Then print bigrams if any
        if bigrams:
            results += "  Bigrams:\n"
            for word, weight in bigrams:
                results += f"    - {word}: {weight:.4f}\n"
        results += "\n"

    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(results)


def main():
    # File paths remain the same
    raw_data_path = '../../data/dataset_combined.csv'
    cleaned_data_path = '../../data/cleaned_dataset_lsa.csv'
    log_file_path = '../../logs/LSA_results.txt'

    os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    download_nltk_resources()

    print("Loading data...")
    if os.path.exists(cleaned_data_path):
        print("Loading preprocessed data...")
        data = pd.read_csv(cleaned_data_path)
        data = data.dropna(subset=['processed_review'])
        data = data[data['processed_review'].str.strip().str.len() > 0]
    else:
        print("Loading raw data...")
        data = pd.read_csv(raw_data_path)
        data = preprocess_data_parallel(data, cleaned_data_path)

    print(f"Dataset size after cleaning: {len(data)} reviews")

    t0 = time()
    svd, vectorizer, X, explained_variance_ratio, common_bigrams = perform_lsa_analysis(
        data, n_components=15  # Reduced from 20 to 15
    )

    save_results(svd, vectorizer, explained_variance_ratio, log_file_path, common_bigrams)

    print(f"Analysis completed in {time() - t0:.2f} seconds")
    print(f"Total variance explained: {explained_variance_ratio.sum():.4f}")


if __name__ == "__main__":
    main()