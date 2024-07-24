from datetime import datetime

import numpy as np
import torch

from scipy import sparse as sp
from scipy.sparse import coo_matrix
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances


class AdjacencyMatrixGenerator:
    STARR_FEAT = [
        "construct to absolute nouns ratio",
        "construct nouns and adjectives percentage",
        "noun to verb ratio",
        "definite_article_percentage",
        "direct object marker percentage",
        "pronouns bound to nouns or verbs percentage",
        "persuasive verb forms (imperative, jussive, cohorative) percentage",
        "preterite percentage",
        "ky percentage",
        "aCr percentage",
        "oM percentage",
        "kya percentage",
        "all conjunctions percentage",
        "non-finite to finite verbs ratio",
        "passive verb forms percentage",
        "total word count",
    ]

    def __init__(
        self,
        vectorizer_type="trigram",
        vectorizer_params=None,
        threshold=0.85,
        distance_metric="cosine",
        normalize=True,
    ):
        self.vectorizer_type = vectorizer_type
        self.vectorizer_params = (
            vectorizer_params if vectorizer_params is not None else {}
        )
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.normalize = normalize

    def _vectorize_text(self, texts):
        if "gram" in self.vectorizer_type:
            return self._vectorize_ngram(texts)
        elif self.vectorizer_type == "tfidf":
            return self._vectorize_tfidf(texts)
        elif self.vectorizer_type == "starr":
            return self._vectorize_starr(texts)
        elif "topic_modeling" in self.vectorizer_type:
            return self._vectorize_topic_modeling(texts)
        else:
            raise ValueError("Unsupported vectorizer type.")

    def _vectorize_ngram(self, texts):
        vectorizer = CountVectorizer(**self.vectorizer_params)
        return vectorizer.fit_transform(texts)

    def _vectorize_tfidf(self, texts):
        vectorizer = TfidfVectorizer(**self.vectorizer_params)
        return vectorizer.fit_transform(texts)

    def _vectorize_starr(self, texts):
        return self.df[self.STARR_FEAT].to_numpy()

    def _vectorize_topic_modeling(self, texts):
        count_vectorizer = CountVectorizer(
            **{"analyzer": "word", "ngram_range": (1, 1), "max_df": 0.5, "min_df": 3}
        )
        count_data = count_vectorizer.fit_transform(texts)

        if self.vectorizer_params.get("type") == "LDA":
            lda = LatentDirichletAllocation(
                n_components=self.vectorizer_params.get("n_topics", 10), random_state=42
            )
            lda_features = lda.fit_transform(count_data)
            return lda_features

        elif self.vectorizer_params.get("type") == "NMF":
            from sklearn.decomposition import NMF

            nmf_model = NMF(
                n_components=self.vectorizer_params.get("n_topics", 10), random_state=42
            )
            nmf_features = nmf_model.fit_transform(count_data)
            return nmf_features
        else:
            raise ValueError("Unsupported topic modeling type.")

    def _compute_adjacency_matrix(self, vectorized_texts):
        if self.distance_metric == "cosine":
            adj_matrix = cosine_similarity(vectorized_texts)
        elif self.distance_metric == "euclidean":
            adj_matrix = euclidean_distances(vectorized_texts)
        elif self.distance_metric == "manhattan":
            adj_matrix = manhattan_distances(vectorized_texts)
        elif self.distance_metric == "jaccard":
            adj_matrix = 1 - squareform(
                pdist(vectorized_texts.toarray(), metric="jaccard")
            )
        else:
            raise ValueError("Unsupported distance metric.")

        return adj_matrix

    def create_adjacency_matrix(self, df):
        if "text" not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column.")
        self.df = df
        texts = df["text"].tolist()
        vectorized_texts = self._vectorize_text(texts)
        adj_matrix = self._compute_adjacency_matrix(vectorized_texts)

        return adj_matrix

    def dilute_adjacency_matrix(self, adj_matrix, threshold):
        if not threshold:
            return adj_matrix

        adj_matrix[adj_matrix < np.quantile(adj_matrix, threshold)] = 0
        return adj_matrix

    def adj_matrix_to_edge_index(self, adj_matrix):
        coo = coo_matrix(adj_matrix)
        edge_index = np.vstack((coo.row, coo.col))
        edge_attr = coo.data
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        return edge_index, edge_attr

    def generate_graph(self, df):
        adj_matrix = self.create_adjacency_matrix(df)
        adj_matrix_filtered = adj_matrix.copy()
        adj_matrix_filtered = self.dilute_adjacency_matrix(
            adj_matrix_filtered, self.threshold
        )
        if self.normalize:
            adj_matrix_filtered = normalize_adj_mat(adj_matrix_filtered)
        print(
            f"{datetime.now()} - {self.vectorizer_type } n edges before filtering: {(adj_matrix != 0).sum()}, n edges after filtering: {(adj_matrix_filtered != 0).sum()}"
        )
        edge_index, edge_attr = self.adj_matrix_to_edge_index(adj_matrix_filtered)
        return edge_index, edge_attr, adj_matrix_filtered


def normalize_adj_mat(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()


class CombinedAdjacencyMatrixGenerator(AdjacencyMatrixGenerator):
    def __init__(
        self, adjacency_generators, combine_method="max", threshold=0.85, normalize=True
    ):
        super().__init__(None, None, threshold)
        self.adjacency_generators = adjacency_generators
        self.combine_method = combine_method
        self.threshold = threshold
        self.normalize = normalize

    def combine_graphs(self, dfs):
        combined_adj_matrix = None

        for generator, df in zip(self.adjacency_generators, dfs):
            _, _, adj_matrix = generator.generate_graph(df)
            if combined_adj_matrix is None:
                combined_adj_matrix = adj_matrix
            else:
                if self.combine_method == "max":
                    combined_adj_matrix = np.maximum(combined_adj_matrix, adj_matrix)
                elif self.combine_method == "add":
                    combined_adj_matrix = combined_adj_matrix + adj_matrix
                elif self.combine_method == "union":
                    combined_adj_matrix = np.logical_or(
                        combined_adj_matrix, adj_matrix
                    ).astype(float)
                else:
                    raise ValueError(
                        "Unsupported combine method. Use 'max', 'add', or 'union'."
                    )

        adj_matrix_filtered = self.dilute_adjacency_matrix(
            combined_adj_matrix, self.threshold
        )
        if self.normalize:
            adj_matrix_filtered = normalize_adj_mat(adj_matrix_filtered)

        edge_index, edge_attr = self.adj_matrix_to_edge_index(adj_matrix_filtered)
        return edge_index, edge_attr, adj_matrix_filtered
