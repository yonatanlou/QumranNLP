import pandas as pd
from scipy import sparse as sp
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    jaccard_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from notebooks.features import get_linkage_matrix
from src.baselines.create_datasets import QumranDataset
import warnings


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


def unsupervised_task(
    dataset: QumranDataset,
    vectorizer_type,
    adjacency_matrix,
    clustering_algo="agglomerative",
):
    vectorizer_matrix = dataset.load_embeddings(vectorizer_type)
    n_clusters = dataset.n_labels
    label_column = dataset.label
    df = dataset.df

    if sp.issparse(vectorizer_matrix):
        vectorizer_matrix = vectorizer_matrix.toarray()
    clusterer = get_clusterer(clustering_algo, n_clusters)

    df["predicted_cluster"] = clusterer.fit_predict(vectorizer_matrix).astype(str)
    le = LabelEncoder()
    le.fit(df[label_column])
    true_labels_encode = le.transform(df[label_column])
    predicted_labels = clusterer.labels_

    ari = adjusted_rand_score(true_labels_encode, predicted_labels)
    # nmi = normalized_mutual_info_score(true_labels_encode, predicted_labels)
    # fmi = fowlkes_mallows_score(true_labels_encode, predicted_labels)
    jaccard = jaccard_score(true_labels_encode, predicted_labels, average="weighted")

    linkage_matrix = get_linkage_matrix(clusterer)
    dasgupta = calculate_dasgupta_score(adjacency_matrix, linkage_matrix)

    metrics = {
        "vectorizer_type": vectorizer_type,
        "ari": ari,
        # "nmi": nmi,
        # "fmi": fmi,
        "jaccard": jaccard,
        "dasgupta": dasgupta,
    }

    return metrics


def evaluate_unsupervised_metrics(
    adjacency_matrix_all, dataset: QumranDataset, vectorizers, path
):
    metrics_list = []
    adjacency_matrix_composition = adjacency_matrix_all[
        dataset.relevant_idx_to_embeddings, :
    ][:, dataset.relevant_idx_to_embeddings]

    for vectorizer in vectorizers:
        metrics = unsupervised_task(
            dataset, vectorizer, adjacency_matrix_composition, "agglomerative"
        )
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list).sort_values(
        by=["dasgupta", "jaccard"], ascending=False
    )
    metrics_df.to_csv(f"{path}/{dataset.label}_unsupervised.csv")
    print(f"Saved metrics to {path}/{dataset.label}_unsupervised.csv")
    return metrics_df


def compute_classification_metrics(model, dataset: QumranDataset, vectorizer_type):
    le = LabelEncoder()
    df = dataset.df
    df["encoded_labels"] = le.fit_transform(df[dataset.label])

    vectorizer_matrix = dataset.load_embeddings(vectorizer_type)
    X_train, y_train = (
        vectorizer_matrix[dataset.train_mask],
        df.loc[dataset.train_mask, "encoded_labels"],
    )
    X_test, y_test = (
        vectorizer_matrix[dataset.test_mask],
        df.loc[dataset.test_mask, "encoded_labels"],
    )
    # Fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    metrics = {
        "model": type(model).__name__,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return metrics


def evaluate_supervised_metrics(models, vectorizers, dataset, path):
    warnings.filterwarnings("ignore")
    metrics_list = []

    for model in tqdm(models, desc="model"):
        for vectorizer_name in vectorizers:
            metrics = compute_classification_metrics(model, dataset, vectorizer_name)
            metrics["vectorizer"] = vectorizer_name
            metrics_list.append(metrics)

    # Convert metrics list to a DataFrame for easier analysis
    metrics_df = pd.DataFrame(metrics_list).sort_values(by="f1_score", ascending=False)
    metrics_df.to_csv(f"{path}/{dataset.label}_supervised.csv", index=False)
    print(f"saved metrics to {path}/{dataset.label}_supervised.csv")
    warnings.filterwarnings("default")
    return metrics_df