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
# Standard CPU UMAP:
from umap.umap_ import UMAP
# For GPU UMAP with cuML (if installed):
# from cuml.manifold import UMAP as cumlUMAP

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize # Keep word_tokenize for splitting sentences if needed
from nltk.stem import WordNetLemmatizer
import re

# --- Gensim Imports ---
from gensim.corpora import Dictionary
from gensim.models import LsiModel, LdaMulticore, Nmf, CoherenceModel, TfidfModel
from gensim.matutils import corpus2dense # To convert sparse corpus to dense matrix for UMAP

import warnings
import psutil
from joblib import Memory
import tempfile

# --- Configuration ---
NUM_TOPICS = 100  # Fixed number of topics
N_CORES_PREPROCESSING = 14 # Cores for text preprocessing (adjust based on your CPU)
N_JOBS_MODELING = 14 # Cores for Gensim LDA/Coherence (-1 might not work same way, specify core count)
VIZ_SAMPLES = 50000 # Max samples for UMAP visualization
CACHE_DIR = os.path.join(tempfile.gettempdir(), 'topic_modeling_cache_gensim')
# Gensim Dictionary Filtering Parameters
NO_BELOW = 10       # Minimum number of documents a word must appear in
NO_ABOVE = 0.6      # Maximum fraction of documents a word can appear in
KEEP_N = 20000      # Keep only the top N most frequent words after filtering
# --- End Configuration ---

os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(location=CACHE_DIR, verbose=0)

# Suppress specific warnings if needed
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) # Gensim sometimes throws these

def print_resource_usage():
    """Prints current CPU and RAM usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory_info = psutil.virtual_memory()
        print(f"CPU Usage: {cpu_percent:.1f}% | "
              f"RAM Usage: {memory_info.percent:.1f}% ({memory_info.used / (1024**3):.2f}/{memory_info.total / (1024**3):.2f} GB)")
    except Exception as e:
        print(f"Could not retrieve resource usage: {e}")

def download_nltk_resources():
    """Downloads necessary NLTK resources."""
    # (pozostaje bez zmian - kod jest taki sam jak w poprzedniej wersji)
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

    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    print("Downloading NLTK resources...")
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Couldn't download NLTK resource '{resource}'. Error: {str(e)}")
            if resource == 'punkt' and 'CERTIFICATE_VERIFY_FAILED' in str(e):
                 print("SSL certificate error. Trying without verification (if possible)...")


def get_custom_stop_words():
    """Returns a list of English stopwords plus custom gaming-related words."""
    # (pozostaje bez zmian - kod jest taki sam jak w poprzedniej wersji)
    gaming_stop_words = {
        'game', 'games', 'play', 'played', 'playing', 'player', 'players',
        'steam', 'review', 'reviews', 'recommend', 'recommended', 'recommendation',
        'hour', 'hours', 'hr', 'hrs',
        'like', 'good', 'bad', 'great', 'best', 'worst', 'really', 'also',
        'much', 'many', 'lot', 'well', 'make', 'made', 'get', 'got',
        'time', 'buy', 'bought', 'price', 'worth', 'money',
        'pc', 'windows', 'early', 'access', 'update', 'release', 'version',
        '10', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', # Numbers often become noise
        'character', 'story', 'level', 'map', 'graphics', 'sound', 'music', # Keep if analyzing aspects, remove if general topics
        'fun', 'enjoy', 'love', 'hate', 'issue', 'problem', 'fix', # Sentiment/common words
        'would', 'could', 'get', 'go', 'one', 'even', 'still', 'since', 'controller', 'keyboard', 'mouse'
    }
    try:
        stop_words_list = stopwords.words('english')
    except LookupError:
        print("NLTK 'stopwords' resource not found. Please run download_nltk_resources() or download manually.")
        print("Using a basic default stopword list.")
        stop_words_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

    return list(set(stop_words_list).union(gaming_stop_words))

# --- Preprocessing functions adjusted for Gensim (output list of tokens) ---

@memory.cache
def process_text_gensim(text, lemmatizer=None, stop_words=None):
    """Processes a single text document for Gensim: tokenizes, removes stopwords/punctuation, lemmatizes.
       Returns a list of tokens."""
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    if stop_words is None:
        stop_words = get_custom_stop_words()

    if not isinstance(text, str):
        return [] # Return empty list for non-string input

    # Lowercase and remove URLs/special characters
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text) # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text) # Keep only letters and spaces

    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens
                         if word not in stop_words and len(word) > 2]

    return lemmatized_tokens # Return list of tokens directly

def process_text_chunk_gensim(chunk, lemmatizer=None, stop_words=None):
    """Processes a chunk of text documents for Gensim."""
    processed_chunk = []
    for text in chunk:
        tokens = process_text_gensim(text, lemmatizer, stop_words)
        if tokens: # Only add if not empty after processing
             processed_chunk.append(tokens)
    return processed_chunk

def preprocess_data_parallel_gensim(data, n_cores=None):
    """Preprocesses text data in parallel for Gensim, returning list of lists of tokens."""
    if n_cores is None:
        n_cores = N_CORES_PREPROCESSING

    n_docs = len(data)
    chunk_size = max(1, min(500, n_docs // (n_cores * 2)))
    num_chunks = (n_docs + chunk_size - 1) // chunk_size
    chunks = [data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

    print(f"Preprocessing {n_docs} documents for Gensim using {n_cores} cores...")
    print(f"Number of chunks: {num_chunks}, Approx. chunk size: {chunk_size}")

    lemmatizer = WordNetLemmatizer()
    stop_words = get_custom_stop_words()

    tokenized_texts = []
    executor = ProcessPoolExecutor(max_workers=n_cores)
    try:
        process_func = partial(process_text_chunk_gensim,
                               lemmatizer=lemmatizer,
                               stop_words=stop_words)
        results = list(tqdm(executor.map(process_func, chunks),
                            total=len(chunks),
                            desc="Processing text chunks"))
    finally:
        executor.shutdown()

    for chunk_result in results:
        tokenized_texts.extend(chunk_result)

    print(f"Preprocessing finished. Kept {len(tokenized_texts)} non-empty documents (tokenized).")
    return tokenized_texts

# --- Coherence Calculation (using Gensim) ---

# @memory.cache # Caching might be complex with Gensim models/dictionaries
def calculate_coherence_gensim(model, tokenized_texts, dictionary, coherence_measure='c_v', top_n=10):
    """Calculates topic coherence using Gensim's CoherenceModel."""
    print(f"Calculating coherence ({coherence_measure})...")
    if not tokenized_texts or not dictionary:
         print("Warning: Empty texts or dictionary provided for coherence calculation.")
         return np.nan
    try:
        coherence_model = CoherenceModel(
            model=model,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence=coherence_measure,
            topn=top_n,
            processes=N_JOBS_MODELING # Use multiple cores if available
        )
        coherence_score = coherence_model.get_coherence()
        print(f"Average Coherence Score ({coherence_measure}): {coherence_score:.4f}")
        return coherence_score
    except Exception as e:
        print(f"Error calculating coherence: {e}")
        # Try getting topics manually if model=model fails (e.g., for LSI)
        try:
            print("Trying manual topic extraction for coherence...")
            topics = []
            num_topics_actual = model.num_topics
            shown_topics = model.show_topics(num_topics=num_topics_actual, num_words=top_n, formatted=False)
            for topic_tuple in shown_topics:
                topics.append([word for word, _ in topic_tuple[1]]) # Extract words

            if not topics:
                 print("Manual topic extraction failed.")
                 return np.nan

            coherence_model_manual = CoherenceModel(
                topics=topics, # Pass extracted topics
                texts=tokenized_texts,
                dictionary=dictionary,
                coherence=coherence_measure,
                topn=top_n,
                processes=N_JOBS_MODELING
            )
            coherence_score = coherence_model_manual.get_coherence()
            print(f"Average Coherence Score ({coherence_measure}) (Manual Topics): {coherence_score:.4f}")
            return coherence_score
        except Exception as e2:
             print(f"Manual coherence calculation also failed: {e2}")
             return np.nan


# --- Helper to get Dense Document-Topic Matrix from Gensim ---

def get_doc_topic_matrix(model, corpus, num_topics):
    """Converts sparse Gensim corpus output to a dense NumPy matrix."""
    print("Converting document-topic distributions to dense matrix...")
    num_docs = len(corpus)
    doc_topic_matrix = np.zeros((num_docs, num_topics), dtype=np.float32)

    # Get topic distributions for all docs
    # model[corpus] yields doc vectors one by one which can be slow
    # Try to get all at once if possible, otherwise iterate
    try:
        # This works directly for LDA typically
        all_topics = model.get_document_topics(corpus, minimum_probability=0.0)
        for i, doc_topics in enumerate(tqdm(all_topics, total=num_docs, desc="Populating matrix")):
             for topic_id, prob in doc_topics:
                 if topic_id < num_topics: # Ensure topic_id is within bounds
                     doc_topic_matrix[i, topic_id] = prob
    except AttributeError:
         # Fallback for models like LSI/NMF that might not have get_document_topics
         print("Model lacks get_document_topics, iterating through corpus...")
         corpus_transformed = model[corpus] # Apply model to entire corpus
         for i, doc_topics in enumerate(tqdm(corpus_transformed, total=num_docs, desc="Populating matrix (iter)"), 0):
              if i >= num_docs: # Safety break
                   print(f"Warning: Index {i} out of bounds for {num_docs} documents.")
                   break
              for topic_id, prob in doc_topics:
                  if topic_id < num_topics:
                      # LSI might return negative values, take absolute? Or normalize later?
                      # Let's store raw value for now. UMAP's cosine metric handles it.
                      doc_topic_matrix[i, topic_id] = prob

    # Normalize rows for LSI/NMF if needed (optional, cosine metric handles scale)
    # For LSI, values aren't probabilities. For NMF they are factors.
    # Normalization might make sense for visualization if scales vary wildly.
    # from sklearn.preprocessing import normalize
    # doc_topic_matrix = normalize(doc_topic_matrix, axis=1, norm='l1') # L1 norm for probability-like

    print(f"Dense document-topic matrix shape: {doc_topic_matrix.shape}")
    return doc_topic_matrix

# --- Visualization (using UMAP on the dense matrix) ---

def visualize_topics_gensim(doc_topic_matrix, method_name, num_topics_viz, save_path):
    """Visualizes topic distributions using UMAP."""
    print(f"Visualizing topics for {method_name} using UMAP...")
    t0 = time()

    max_samples = min(VIZ_SAMPLES, doc_topic_matrix.shape[0])
    if doc_topic_matrix.shape[0] > max_samples:
        print(f"Sampling {max_samples} documents for visualization...")
        indices = np.random.choice(doc_topic_matrix.shape[0], max_samples, replace=False)
        doc_topics_sampled = doc_topic_matrix[indices]
    else:
        doc_topics_sampled = doc_topic_matrix

    # Handle potential all-zero rows after sampling (if docs had no assigned topics)
    non_zero_rows = np.abs(doc_topics_sampled).sum(axis=1) > 1e-6
    if not np.all(non_zero_rows):
        print(f"Warning: Removing {np.sum(~non_zero_rows)} documents with near-zero topic assignments before UMAP.")
        doc_topics_sampled = doc_topics_sampled[non_zero_rows]
        if doc_topics_sampled.shape[0] < 5: # Need at least a few points for UMAP
             print("Error: Too few documents remain after filtering zero rows. Skipping visualization.")
             return


    print("Running UMAP dimensionality reduction...")
    try:
        # For GPU UMAP with cuML:
        # umap_model = cumlUMAP(n_components=2, n_neighbors=15, min_dist=0.1,
        #                       metric='cosine', random_state=42, verbose=True)
        # Standard CPU UMAP:
        umap_model = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine', # Cosine works well for topic distributions/vectors
            random_state=42,
            n_jobs=N_JOBS_MODELING,
            verbose=True
        )
        # Ensure input is C-contiguous and float32 for UMAP
        umap_input = np.ascontiguousarray(doc_topics_sampled, dtype=np.float32)
        umap_output = umap_model.fit_transform(umap_input)
        print(f"UMAP finished in {time() - t0:.2f} seconds.")

    except Exception as e:
        print(f"UMAP failed: {e}")
        # Provide more detail if possible
        if "negative" in str(e).lower() and method_name == 'LSA':
            print("Hint: UMAP might struggle with negative values from LSA if not handled (e.g., abs() or different metric). Cosine metric should be okay.")
        print("Skipping visualization.")
        return

    # Determine dominant topic based on the highest score/probability
    dominant_topics = np.argmax(doc_topics_sampled, axis=1)

    plt.figure(figsize=(14, 9))
    plt.style.use('seaborn-v0_8-darkgrid')

    scatter = plt.scatter(umap_output[:, 0],
                          umap_output[:, 1],
                          c=dominant_topics,
                          cmap='Spectral',
                          alpha=0.6,
                          s=10)

    plt.colorbar(scatter, label='Dominant Topic ID')
    plt.title(f'{method_name} Topic Distribution ({num_topics_viz} topics, UMAP visualization)', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    info_text = f'Topics: {num_topics_viz}\nSamples: {doc_topics_sampled.shape[0]}'
    plt.figtext(0.01, 0.01, info_text, fontsize=9, alpha=0.7)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {save_path}")
    print("-" * 30)

# --- Training and Evaluation Function using Gensim ---

def train_and_evaluate_gensim(tokenized_texts, method):
    """Trains a Gensim topic model (LSA, LDA, or NMF) and returns results."""
    print(f"\n--- Starting Gensim {method.upper()} Analysis ({NUM_TOPICS} topics) ---")
    print_resource_usage()
    t_start = time()

    # 1. Create Dictionary and Corpus
    print("Creating Gensim Dictionary and Corpus...")
    dictionary = Dictionary(tokenized_texts)
    print(f"Original dictionary size: {len(dictionary)}")
    # Filter extremes
    dictionary.filter_extremes(no_below=NO_BELOW, no_above=NO_ABOVE, keep_n=KEEP_N)
    dictionary.compactify() # Remove gaps in ids
    print(f"Filtered dictionary size: {len(dictionary)}")

    # Create BoW Corpus (needed for all models initially)
    corpus = [dictionary.doc2bow(text) for text in tqdm(tokenized_texts, desc="Creating BoW Corpus")]

    # Create TF-IDF Corpus (only for LSA and NMF)
    corpus_tfidf = None
    if method in ['lsa', 'nmf']:
        print("Creating TF-IDF Corpus...")
        tfidf_model = TfidfModel(corpus, id2word=dictionary)
        corpus_tfidf = tfidf_model[corpus]

    print(f"Dictionary and Corpus creation time: {time() - t_start:.2f}s")
    print_resource_usage()
    gc.collect()

    # 2. Model Training
    print(f"Training Gensim {method.upper()} model...")
    t_model_start = time()
    model = None
    train_corpus = corpus_tfidf if method in ['lsa', 'nmf'] else corpus

    if not train_corpus:
        print("Error: Training corpus is empty. Skipping model training.")
        return None

    try:
        if method == 'lsa':
            model = LsiModel(
                train_corpus,
                num_topics=NUM_TOPICS,
                id2word=dictionary,
                chunksize=2000, # Process docs in chunks
                # power_iters=2, # Number of power iterations
                # extra_samples=100 # Extra samples for accuracy
            )
        elif method == 'lda':
            model = LdaMulticore(
                corpus=train_corpus, # LDA uses BoW counts
                num_topics=NUM_TOPICS,
                id2word=dictionary,
                workers=N_JOBS_MODELING,
                chunksize=2000,     # Documents processed in each worker pass
                passes=15,          # Increase passes for potentially better convergence
                iterations=100,     # Iterations per document pass
                eval_every=None,   # Disable perplexity evaluation during training for speed
                random_state=42
            )
        elif method == 'nmf':
             model = Nmf(
                 corpus=train_corpus, # NMF uses TF-IDF here
                 num_topics=NUM_TOPICS,
                 id2word=dictionary,
                 chunksize=2000,
                 passes=15,          # Number of full passes over the corpus
                 eval_every=10,      # Evaluate reconstruction error periodically
                 random_state=42,
                 # W_update_rho=0.1, # Learning rate for W (optional)
                 # H_update_rho=0.1, # Learning rate for H (optional)
             )

        print(f"Gensim {method.upper()} model training time: {time() - t_model_start:.2f}s")
        print_resource_usage()
    except Exception as e:
        print(f"ERROR: Failed to train Gensim {method.upper()} model: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        return None # Return None if training fails

    # 3. Coherence Calculation
    coherence_score = calculate_coherence_gensim(model, tokenized_texts, dictionary, coherence_measure='c_v')

    # 4. Get Dense Document-Topic Matrix for visualization/saving
    doc_topic_matrix = get_doc_topic_matrix(model, train_corpus, NUM_TOPICS)

    # 5. Prepare Results
    results = {
        'method': method,
        'model': model,
        'dictionary': dictionary,
        'corpus': train_corpus, # Store the corpus used for training
        'doc_topic_matrix': doc_topic_matrix, # Dense matrix
        'coherence': coherence_score,
        'total_time': time() - t_start
    }

    print(f"--- Gensim {method.upper()} Analysis Finished (Total Time: {results['total_time']:.2f}s) ---")
    gc.collect()
    return results

# --- Saving Results Function for Gensim ---

def save_results_gensim(results_dict, log_file_path):
    """Saves Gensim model results (top words per topic, coherence) and triggers visualization."""
    if results_dict is None:
        print("Skipping saving results because training failed.")
        return

    method = results_dict['method']
    model = results_dict['model']
    # dictionary = results_dict['dictionary'] # Not directly needed for saving top words
    doc_topic_matrix = results_dict['doc_topic_matrix']
    coherence = results_dict['coherence']
    num_topics_actual = model.num_topics # Get actual number of topics from model

    print(f"\nSaving Gensim {method.upper()} results to {log_file_path}...")

    # Prepare results string
    output = f"--- Gensim {method.upper()} Topic Modeling Results ---\n"
    output += f"Number of topics: {num_topics_actual}\n"
    output += f"Coherence Score (c_v): {coherence:.4f}\n"
    output += f"Dictionary Size (filtered): {len(results_dict['dictionary'])}\n"
    output += f"Total analysis time: {results_dict['total_time']:.2f} seconds\n\n"
    output += "--- Top Words per Topic ---\n\n"

    # Determine topic importance/order (optional, using the dense matrix)
    topic_prevalence = np.sum(np.abs(doc_topic_matrix), axis=0) # Use abs for LSI potentially
    # Ensure prevalence array matches number of topics
    if len(topic_prevalence) != num_topics_actual:
         print(f"Warning: Prevalence array length ({len(topic_prevalence)}) doesn't match model topics ({num_topics_actual}). Using direct topic order.")
         topic_order = range(num_topics_actual)
    else:
         topic_order = np.argsort(-topic_prevalence) # Sort topics by prevalence (descending)


    # Get top words using model's method
    try:
        shown_topics = model.show_topics(num_topics=num_topics_actual, num_words=20, formatted=True)
        # Create a dictionary for easy lookup by topic ID
        topic_dict = {int(t[0]): t[1] for t in shown_topics}

        for i, topic_idx in enumerate(topic_order):
            if topic_idx in topic_dict:
                output += f"Topic {i+1} (Original ID: {topic_idx}, Prevalence Score: {topic_prevalence[topic_idx]:.2f}):\n"
                # Format the string like "word1 (weight1), word2 (weight2)..."
                words_weights = topic_dict[topic_idx].split(' + ')
                for item in words_weights:
                    try:
                        weight, word = item.split('*')
                        word = word.strip().replace('"', '') # Clean up word
                        weight = float(weight.strip())
                        output += f"  - {word} ({weight:.4f})\n"
                    except ValueError:
                        output += f"  - Error parsing: {item}\n" # Handle potential parsing issues
                output += "\n"
            else:
                 output += f"Topic {i+1} (Original ID: {topic_idx}, Prevalence Score: {topic_prevalence[topic_idx]:.2f}):\n"
                 output += "  - (Could not retrieve formatted words for this topic)\n\n"

    except Exception as e:
        output += f"Error retrieving or formatting topics: {e}\n"
        print(f"Error retrieving/formatting topics: {e}")


    # Write to file
    try:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Results saved successfully.")
    except Exception as e:
        print(f"ERROR: Could not write results to {log_file_path}. Error: {e}")

    # Generate and save visualization
    viz_path = log_file_path.replace('.txt', '_visualization.png')
    visualize_topics_gensim(doc_topic_matrix, method.upper(), num_topics_actual, viz_path)

# --- Main Function ---

def main():
    """Main function to run the Gensim topic modeling pipeline."""
    data_path = '../../data/dataset_combined.csv' # Adjust path as needed
    results_dir = '../../logs_gensim/' # Directory for output files
    results_path_lsa = os.path.join(results_dir, 'gensim_LSA_results.txt')
    results_path_lda = os.path.join(results_dir, 'gensim_LDA_results.txt')
    results_path_nmf = os.path.join(results_dir, 'gensim_NMF_results.txt')

    print("--- Starting Gensim Topic Modeling Pipeline ---")
    print(f"Number of topics to find: {NUM_TOPICS}")
    print(f"Using {N_CORES_PREPROCESSING} cores for preprocessing.")
    print(f"Using up to {N_JOBS_MODELING} cores for modeling (LDA/Coherence).")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Dictionary filtering: no_below={NO_BELOW}, no_above={NO_ABOVE}, keep_n={KEEP_N}")


    print("\nInitial resource usage:")
    print_resource_usage()

    os.makedirs(results_dir, exist_ok=True)
    download_nltk_resources()

    print("\nLoading data...")
    try:
        data = pd.read_csv(data_path, usecols=['review'], low_memory=False)
        data.dropna(subset=['review'], inplace=True)
        reviews = data['review'].astype(str).tolist()
        print(f"Loaded {len(reviews)} reviews.")
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_path}")
        return
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        return

    del data
    gc.collect()

    print("\nPreprocessing texts for Gensim (outputting token lists)...")
    t_prep_start = time()
    tokenized_texts = preprocess_data_parallel_gensim(reviews, n_cores=N_CORES_PREPROCESSING)
    print(f"Preprocessing time: {time() - t_prep_start:.2f}s")
    del reviews
    gc.collect()
    print_resource_usage()

    if not tokenized_texts:
        print("ERROR: No texts remained after preprocessing. Exiting.")
        return

    # --- Run Analyses (Gensim) ---
    # Note: Dictionary/Corpus are created inside train_and_evaluate_gensim

    # LSA (LSI)
    lsa_results = train_and_evaluate_gensim(tokenized_texts, 'lsa')
    if lsa_results:
        save_results_gensim(lsa_results, results_path_lsa)
        del lsa_results # Free memory
        gc.collect()

    # LDA
    lda_results = train_and_evaluate_gensim(tokenized_texts, 'lda')
    if lda_results:
        save_results_gensim(lda_results, results_path_lda)
        del lda_results # Free memory
        gc.collect()

    # NMF
    nmf_results = train_and_evaluate_gensim(tokenized_texts, 'nmf')
    if nmf_results:
        save_results_gensim(nmf_results, results_path_nmf)
        del nmf_results # Free memory
        gc.collect()

    print("\n--- Gensim Pipeline Finished ---")
    print("Final resource usage:")
    print_resource_usage()
    # Clear cache if desired (optional)
    # print("Clearing cache...")
    # memory.clear()

if __name__ == "__main__":
    # Set spawn method for multiprocessing compatibility (important on Windows/macOS)
    # Needs to be done *before* any multiprocessing pools are created
    if multiprocessing.get_start_method(allow_none=True) is None:
         multiprocessing.set_start_method('spawn', force=True)
    elif multiprocessing.get_start_method(allow_none=True) != 'spawn':
         # If already set but not to 'spawn', force it if needed, be cautious
         try:
              multiprocessing.set_start_method('spawn', force=True)
              print("Forcing multiprocessing start method to 'spawn'.")
         except RuntimeError as e:
              print(f"Could not force multiprocessing start method to 'spawn': {e}")
              print("Proceeding with current method:", multiprocessing.get_start_method())

    main()