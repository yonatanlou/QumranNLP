import os
import pickle

from notebooks.constants import BERT_MODELS
from notebooks.features import vectorize_text


def load_vectorizers(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            processed_vectorizers = pickle.load(f)
            print(f"Loaded the embeddings: {list(processed_vectorizers.keys())}")
    else:
        processed_vectorizers = {}
    return processed_vectorizers


def save_vectorizers(path, vectorizers):
    with open(path, "wb") as f:
        pickle.dump(vectorizers, f)


def get_vectorizer_types():
    return BERT_MODELS + ["tfidf", "trigram", "starr"]


def process_vectorizer(df, vectorizer_type, processed_vectorizers):
    if vectorizer_type in processed_vectorizers:
        return processed_vectorizers
    else:
        X = vectorize_text(df, "text", vectorizer_type)
        processed_vectorizers[vectorizer_type] = X
        return processed_vectorizers


def load_or_genereate_embeddings(df, path, vectorizers):
    processed_vectorizers = load_vectorizers(path)

    for vectorizer_type in vectorizers:
        processed_vectorizers = process_vectorizer(
            df, vectorizer_type, processed_vectorizers
        )

    save_vectorizers(path, processed_vectorizers)
    print(f"loaded the following embeddings {processed_vectorizers.keys()}")
    return processed_vectorizers
