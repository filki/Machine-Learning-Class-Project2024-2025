import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re
from time import time
import multiprocessing
from joblib import Parallel, delayed
import gc
import warnings
warnings.filterwarnings('ignore')


def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)


def clean_text(text):
    if not isinstance(text, str):
        return ''

    # Compile regex patterns once
    url_pattern = re.compile(r'http\S+|www\S+|https\S+')
    number_pattern = re.compile(r'(?<!\d)[0-9]+(?!\d)')
    special_char_pattern = re.compile(r'[^\w\s\']')

    # Apply patterns
    text = url_pattern.sub('', text)
    text = number_pattern.sub('', text)
    text = special_char_pattern.sub(' ', text)
    text = ' '.join(text.split())
    return text.lower()


def get_custom_stop_words():
    gaming_stop_words = {
        'game', 'games', 'play', 'played', 'playing',
        'steam', 'review', 'reviews', 'recommend',
        'recommended', 'hour', 'hours', 'like', 'good',
        'bad', 'great', 'best', 'worst', 'really',
        'much', 'many', 'lot', 'well', 'make', 'made'
    }
    return set(stopwords.words('english')).union(gaming_stop_words)


def process_chunk(texts, lemmatizer, stop_words):
    return [preprocess_text(text, lemmatizer, stop_words) for text in texts]


def preprocess_text(text, lemmatizer, stop_words):
    if not isinstance(text, str):
        return ''

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens
              if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def preprocess_data_parallel(data, cleaned_data_path, chunk_size=10000):
    print("Starting parallel preprocessing...")

    # Remove NaN values from review column first
    print("Removing NaN values...")
    data = data.dropna(subset=['review'])

    # Clean texts in parallel
    print("Cleaning texts...")
    with Parallel(n_jobs=-1, prefer="threads") as parallel:
        data['cleaned_review'] = parallel(
            delayed(clean_text)(text)
            for text in data['review']
        )

    # Initialize resources for text processing
    lemmatizer = WordNetLemmatizer()
    stop_words = get_custom_stop_words()

    # Process in chunks to manage memory
    processed_reviews = []
    n_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size else 0)

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data))
        chunk = data['cleaned_review'][start_idx:end_idx]

        # Process chunk in parallel
        with Parallel(n_jobs=-1, prefer="threads") as parallel:
            processed_chunk = parallel(
                delayed(preprocess_text)(text, lemmatizer, stop_words)
                for text in chunk
            )
        processed_reviews.extend(processed_chunk)

        print(f"Processed chunk {i + 1}/{n_chunks} ({end_idx}/{len(data)} reviews)")
        gc.collect()

    data['processed_review'] = processed_reviews

    # Remove empty reviews and NaN values
    print("Cleaning up processed data...")
    data = data[data['processed_review'].str.len() > 0]
    data = data.dropna(subset=['processed_review'])

    print(f"Final dataset size: {len(data)} reviews")
    print(f"Saving cleaned data to: {cleaned_data_path}")
    data.to_csv(cleaned_data_path, index=False)
    return data


def perform_lda_analysis(data, n_topics=300, batch_size=4096):
    print("Preparing data for vectorization...")

    # Ensure no NaN values before vectorization
    data = data.dropna(subset=['processed_review'])

    # Remove empty strings and whitespace-only strings
    data = data[data['processed_review'].str.strip().str.len() > 0]

    print(f"Number of valid documents for analysis: {len(data)}")

    print("Creating document-term matrix...")
    vectorizer = CountVectorizer(
        max_features=15000,
        min_df=10,
        max_df=0.90,
        token_pattern=r'\b\w+\b',
        lowercase=False
    )

    print("Vectorizing documents...")
    X = vectorizer.fit_transform(data['processed_review'])
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    print("Training LDA model...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method='online',
        batch_size=batch_size,
        max_iter=10,
        learning_decay=0.7,
        random_state=42,
        n_jobs=-1
    )

    lda.fit(X)

    return lda, vectorizer, X


def save_results(lda, vectorizer, log_file_path, perplexity=None):
    feature_names = vectorizer.get_feature_names_out()
    results = f"\n{'=' * 30}\nNumber of topics: {lda.n_components}\n"
    if perplexity:
        results += f"Model perplexity: {perplexity:.2f}\n"
    results += "Topics:\n"

    for idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-15 - 1:-1]  # Get top 15 words
        top_words = [feature_names[i] for i in top_words_idx]
        word_weights = [topic[i] for i in top_words_idx]

        topic_str = f"Topic {idx + 1}:\n"
        for word, weight in zip(top_words, word_weights):
            topic_str += f"  - {word}: {weight:.4f}\n"
        results += topic_str + "\n"

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write(results)

def main():
    # File paths
    raw_data_path = '../../data/dataset_combined.csv'
    cleaned_data_path = '../../data/cleaned_dataset.csv'
    log_file_path = '../../logs/LDA_results.txt'

    # Create directories
    os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Download NLTK resources
    download_nltk_resources()

    # Load data
    print("Loading data...")
    if os.path.exists(cleaned_data_path):
        print("Loading preprocessed data...")
        data = pd.read_csv(cleaned_data_path)
        # Verify data quality even for preprocessed data
        data = data.dropna(subset=['processed_review'])
        data = data[data['processed_review'].str.strip().str.len() > 0]
    else:
        print("Loading raw data...")
        data = pd.read_csv(raw_data_path)
        data = preprocess_data_parallel(data, cleaned_data_path)

    print(f"Dataset size after cleaning: {len(data)} reviews")

    # Perform LDA
    t0 = time()
    lda, vectorizer, X = perform_lda_analysis(data)

    # Calculate perplexity
    perplexity = lda.perplexity(X)

    # Save results
    save_results(lda, vectorizer, log_file_path, perplexity)

    print(f"Analysis completed in {time() - t0:.2f} seconds")
    print(f"Final perplexity: {perplexity:.2f}")


if __name__ == "__main__":
    main()