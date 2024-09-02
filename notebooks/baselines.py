import pandas as pd

from config import BASE_DIR
from notebooks.constants import BERT_MODELS

# This file will construct 6 baselines:
# 1. Sectarian - Supervised and unsupervised
# 2. Composition - Supervised and unsupervised
# 3. Scroll - Supervised and unsupervised


CONTEXT_SIMILIARITY_WINDOW = 3
CHUNK_SIZE = 100
DATA_PATH = f"{BASE_DIR}/notebooks/data/filtered_text_and_starr_features_{CHUNK_SIZE}_words_nonbib_17_06_2024.csv"
df = pd.read_csv(DATA_PATH)
df["original_index"] = range(len(df))
import pickle
from notebooks.features import (
    vectorize_text,
    get_linkage_matrix,
)
from src.baselines.utils import create_adjacency_matrix, set_seed_globally
import os


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


set_seed_globally()
processed_vectorizers_path = f"{BASE_DIR}/notebooks/data/processed_vectorizers.pkl"
processed_vectorizers = load_vectorizers(processed_vectorizers_path)

for vectorizer_type in get_vectorizer_types():
    process_vectorizer(df, vectorizer_type, processed_vectorizers)

save_vectorizers(processed_vectorizers_path, processed_vectorizers)

idx_to_remove_composition = df["composition"].isna()
idx_to_remove_sectarian = (df["section"].isna()) | (df["section"] == "unknown")
processed_vectorizers_comp, processed_vectorizers_sec = {}, {}

for vec_type, vectorizer_mat in processed_vectorizers.items():
    processed_vectorizers_comp[vec_type] = vectorizer_mat[~idx_to_remove_composition]
    processed_vectorizers_sec[vec_type] = vectorizer_mat[~idx_to_remove_sectarian]

from sklearn.cluster import AgglomerativeClustering, DBSCAN
import plotly.express as px
from warnings import simplefilter
from scipy import sparse as sp
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    jaccard_score,
)
from sklearn.preprocessing import LabelEncoder


def get_clusterer(clustering_algo, n_clusters):
    # Apply the selected clustering algorithm
    if clustering_algo == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    elif clustering_algo == "agglomerative":
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters, compute_distances=True, linkage="ward"
        )
    elif clustering_algo == "dbscan":
        clusterer = DBSCAN(eps=6, min_samples=5)
    else:
        raise ValueError(
            "Unsupported clustering algorithm. Choose from 'kmeans', 'agglomerative', or 'dbscan'."
        )
    return clusterer


from sknetwork.hierarchy import dasgupta_score as calculate_dasgupta_score


def compute_metrics(
    processed_vectorizers,
    vectorizer_type,
    df,
    label_column,
    adjacency_matrix,
    clustering_algo="kmeans",
    n_clusters=2,
):
    vectorizer_matrix = processed_vectorizers.get(vectorizer_type)
    if sp.issparse(vectorizer_matrix):
        vectorizer_matrix = vectorizer_matrix.toarray()
    clusterer = get_clusterer(clustering_algo, n_clusters)

    df["predicted_cluster"] = clusterer.fit_predict(vectorizer_matrix).astype(
        str
    )  # Convert labels to strings

    # Compute evaluation metrics
    le = LabelEncoder()
    le.fit(df[label_column])
    true_labels_encode = le.transform(df[label_column])
    predicted_labels = clusterer.labels_

    ari = adjusted_rand_score(true_labels_encode, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels_encode, predicted_labels)
    fmi = fowlkes_mallows_score(true_labels_encode, predicted_labels)
    jaccard = jaccard_score(true_labels_encode, predicted_labels, average="weighted")

    linkage_matrix = get_linkage_matrix(clusterer)
    dasgupta = calculate_dasgupta_score(adjacency_matrix, linkage_matrix)

    metrics = {
        "vectorizer_type": vectorizer_type,
        "ari": ari,
        "nmi": nmi,
        "fmi": fmi,
        "jaccard": jaccard,
        "dasgupta": dasgupta,
    }

    return metrics


adjacency_matrix = create_adjacency_matrix(
    df,
    context_similiarity_window=CONTEXT_SIMILIARITY_WINDOW,
    composition_level=True,
)
adjacency_matrix_comp = adjacency_matrix[~idx_to_remove_composition, :][
    :, ~idx_to_remove_composition
]
adjacency_matrix_sec = adjacency_matrix[~idx_to_remove_sectarian, :][
    :, ~idx_to_remove_sectarian
]

df_comp = df[~idx_to_remove_composition]
vectorizer_types = processed_vectorizers_comp.keys()
metrics_list = []

for vectorizer_type in vectorizer_types:
    metrics = compute_metrics(
        processed_vectorizers_comp,
        vectorizer_type,
        df_comp,
        "composition",
        adjacency_matrix_comp,
        clustering_algo="agglomerative",
        n_clusters=df_comp["composition"].nunique(),
    )
    metrics_list.append(metrics)

# Convert metrics list to a DataFrame for easier analysis
metrics_df = pd.DataFrame(metrics_list).sort_values(
    by=["dasgupta", "jaccard"], ascending=False
)
PATH = f"{BASE_DIR}/notebooks/reports/baselines_test/unsupervised_clustering_composition_level"
if not os.path.exists(PATH):
    os.makedirs(PATH)
metrics_df.to_csv(f"{PATH}/unsupervised_clustering_composition_level.csv", index=False)


# def generate_plots(
#         processed_vectorizers,
#         vectorizer_type,
#         df,
#         label_column,
#         clustering_algo="kmeans",
#         n_clusters=2,
# ):
#     simplefilter(action="ignore", category=FutureWarning)
#     vectorizer_matrix = processed_vectorizers.get(vectorizer_type)
#     if sp.issparse(vectorizer_matrix):
#         vectorizer_matrix = vectorizer_matrix.toarray()
#
#     clusterer = get_clusterer(clustering_algo, n_clusters)
#     df["predicted_cluster"] = clusterer.fit_predict(vectorizer_matrix).astype(
#         str
#     )  # Convert labels to strings if needed
#
#     # Apply t-SNE for dimensionality reduction
#     tsne = TSNE(n_components=2, random_state=42)
#     tsne_results = tsne.fit_transform(vectorizer_matrix)
#
#     # Add t-SNE results to the dataframe
#     df["tsne-2d-one"] = tsne_results[:, 0]
#     df["tsne-2d-two"] = tsne_results[:, 1]
#
#     # Create a facet plot to compare true and predicted labels
#     df["label_type"] = "Predicted"
#     true_labels_df = df.copy()
#     true_labels_df["predicted_cluster"] = true_labels_df[label_column]
#     true_labels_df["label_type"] = "True"
#     combined_df = pd.concat([df, true_labels_df])
#
#     facet_fig = px.scatter(
#         combined_df,
#         x="tsne-2d-one",
#         y="tsne-2d-two",
#         color="predicted_cluster",
#         facet_col="label_type",
#         hover_data={"book": True, label_column: True, "sentence_path": True},
#         title=f"t-SNE visualization of clusters: Predicted vs. True ({vectorizer_type}, {clustering_algo})",
#     )
#
#     # Update the figure size
#     facet_fig.update_layout(
#         width=1000,  # Adjust the width as needed
#         height=800,  # Adjust the height as needed
#     )
#
#     facet_fig.update_traces(marker=dict(size=10), selector=dict(mode="markers+text"))
#     simplefilter(action="default", category=FutureWarning)
#     return facet_fig
