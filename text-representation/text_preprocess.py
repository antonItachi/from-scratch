import numpy as np
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter


stop_words = ENGLISH_STOP_WORDS


def tokenizer(text):
    combined_text = str(text).lower()
    combined_text = re.sub(r'\d+', '', combined_text)  # Remove numbers
    combined_text = re.sub(r"[^\w\s']+", '', combined_text)  # Remove punctuation (except apostrophes)
    combined_text = re.sub(r"\s+", " ", combined_text)  # Remove extra spaces
    combined_text = combined_text.strip()
    tokens = combined_text.split(' ')
    tokens = [word for word in tokens if len(word) > 1 and word not in stop_words]  # Remove stop words and empty tokens
    return tokens


def decode(tokens, word_to_idx):
    """
    word_to_idx is dict where keys are unique words and values are their personal indexes
    """
    # Receiving TimeSeries and preprcccess every row separetly
    decoded_list = []

    # now we can assign for each token in text their id according to our vocabulary
    for token in tokens:
        decoded_list.append(word_to_idx[token])
    return decoded_list


def bag_of_words(tokens):
    # only by 1 row ( 1 peace of text or 1 document)
    freq = Counter()

    for token in tokens:
        freq[token] += 1

    return freq


def calculate_term_frequencies(tokens):
    """Calculate term frequencies (TF) for the given tokens."""
    freq = Counter(tokens)
    total_terms = len(tokens)
    return {key: value / total_terms for key, value in freq.items()}


def calculate_document_frequencies(docs):
    """Calculate document frequencies (DF) for the given documents."""
    df = Counter()
    for tokens in docs:
        unique_terms = set(tokens)
        for term in unique_terms:
            df[term] += 1
    return df


def calculate_idf(df, N):
    return {term: np.log(N / (df[term] + 1)) for term in df}


def calculate_tf_idf(docs):
    N = len(docs)
    all_tfs = []
    df = calculate_document_frequencies(docs)
    idf = calculate_idf(df, N)

    for tokens in docs:
        tf = calculate_term_frequencies(tokens)
        tf_idf = {term: tf[term] * idf[term] for term in tf}
        all_tfs.append(tf_idf)

    return all_tfs


def normalize_vector_shape(tf_idf_results, vocab_size):

    tf_idf_matrix = np.zeros((len(tf_idf_results), vocab_size))

    for i, doc in enumerate(tf_idf_results):
        for token, tfidf_value in doc.items():
            tf_idf_matrix[i, token] = tfidf_value
    return tf_idf_matrix