import os
import pickle
import pandas as pd

from src.constants import BERT_MODELS
from notebooks.features import vectorize_text


class VectorizerProcessor:
    """
    A class to load, process, and save text vectorizers.

    Attributes:
        df (pd.DataFrame): DataFrame containing the text data.
        path (str): Path to save/load the vectorizers.
        vectorizers (list): List of vectorizer types to be processed.
        reprocess_vectorizers (list): List of vectorizers to be reprocessed.
        processed_vectorizers (dict): Dictionary storing processed vectorizers.
    """

    def __init__(
        self, df: pd.DataFrame, path: str, vectorizers: list, reprocess_vectorizers=None
    ):
        if reprocess_vectorizers is None:
            reprocess_vectorizers = []
        self.df = df
        self.path = path
        self.vectorizers = vectorizers
        self.reprocess_vectorizers = reprocess_vectorizers
        self.processed_vectorizers = self.load_vectorizers()

    def load_vectorizers(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                processed_vectorizers = pickle.load(f)
                print(f"Loaded the embeddings: {list(processed_vectorizers.keys())}")
        else:
            processed_vectorizers = {}
        return processed_vectorizers

    def save_vectorizers(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.processed_vectorizers, f)

    def process_vectorizer(self, vectorizer_type, reprocess=False):
        if vectorizer_type in self.processed_vectorizers and not reprocess:
            return
        else:
            print(f"Processing {vectorizer_type}...")
            X = vectorize_text(self.df, "text", vectorizer_type)
            self.processed_vectorizers[vectorizer_type] = X

    def load_or_generate_embeddings(self):
        reprocess = False
        for vectorizer_type in self.vectorizers:
            if vectorizer_type in self.reprocess_vectorizers:
                reprocess = True
            self.process_vectorizer(vectorizer_type, reprocess=reprocess)

        self.save_vectorizers()
        print(f"Loaded the following embeddings {self.processed_vectorizers.keys()}")
        return self.processed_vectorizers


def get_vectorizer_types():
    return BERT_MODELS + ["tfidf", "trigram", "starr"]
