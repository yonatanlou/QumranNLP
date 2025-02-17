import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics.pairwise import cosine_similarity
from config import BASE_DIR
from config import get_paths_by_domain

import scipy
from src.plots_generation.analysis_utils import (
    generate_dendrogram_plot,
    cluster_and_get_metrics,
    get_gae_embeddings,
)
from src.baselines.utils import (
    set_seed_globally,
)
from src.baselines.embeddings import VectorizerProcessor, get_bert_models


def hirerchial_clustering_by_sectarian(
    df: pd.DataFrame,
    title,
    embeddings,
    label_to_plot,
    kwargs,
    path_to_save=None,
):
    linkage_m = kwargs["linkage_m"]
    set_seed_globally()
    df = df.reset_index()
    # get labels
    df_labeled_for_clustering = df
    df_labeled_for_clustering["label"] = df_labeled_for_clustering["section"]
    # get embeddings
    label_idxs = list(df_labeled_for_clustering.index)
    embeddings_tmp = embeddings[label_idxs]
    if isinstance(embeddings_tmp, scipy.sparse.csr_matrix):
        embeddings_tmp = embeddings_tmp.toarray()

    # get metrics and linkage
    linkage_matrix, metrics = cluster_and_get_metrics(
        df_labeled_for_clustering, embeddings_tmp, None, linkage_m
    )

    # plot
    if label_to_plot:
        generate_dendrogram_plot(
            df_labeled_for_clustering,
            linkage_matrix,
            title,
            metrics,
            label_to_plot,
            path_to_save,
            kwargs,
        )
    return metrics


def average_embeddings_by_scroll(
    X, df_no_nulls, groupby_col="composition", outlier_threshold=3
):
    unique_books = df_no_nulls.dropna(subset=[groupby_col], axis=0)[
        groupby_col
    ].unique()
    averaged_embeddings = []
    new_df = pd.DataFrame()

    for book in unique_books:
        # Get the embeddings for the current book
        book_mask = df_no_nulls[groupby_col] == book
        book_indices = np.where(book_mask)[0]
        book_embeddings = X[book_indices]

        # Compute the initial average embedding
        avg_embedding = np.mean(book_embeddings, axis=0, keepdims=True)

        # Calculate cosine similarities of each embedding to the average
        similarities = cosine_similarity(book_embeddings, avg_embedding).flatten()

        # Compute Z-scores for the cosine similarities
        similarity_z_scores = zscore(similarities)

        # Filter out embeddings with Z-scores above the threshold
        valid_indices = np.abs(similarity_z_scores) <= outlier_threshold
        valid_embeddings = book_embeddings[valid_indices]
        print(
            "Removed",
            book_indices.shape[0] - valid_embeddings.shape[0],
            "outliers out of",
            book_indices.shape[0],
            f" for {book}",
        )
        n_chunks = valid_embeddings.shape[0]

        # If no valid embeddings remain after outlier removal, skip this group
        if valid_embeddings.size == 0:
            continue

        # Recompute the average of valid embeddings
        avg_embedding = np.mean(valid_embeddings, axis=0)
        averaged_embeddings.append(avg_embedding)

        # Add book info to new DataFrame
        book_df = df_no_nulls[book_mask].head(1).copy()
        book_df["n_chunks"] = n_chunks
        new_df = pd.concat([new_df, book_df], ignore_index=True)

        # Convert averaged embeddings to numpy array
    averaged_embeddings = np.array(averaged_embeddings)
    return averaged_embeddings, new_df


def format_composition(value):
    if value == "CD":
        return value
    formatted_value = value.replace("_", " ").replace(".", " ")
    return formatted_value.title()


if __name__ == "__main__":
    exp_dir = f"{BASE_DIR}/experiments/dss/sectarian_similarities"
    DOMAIN = "dss"
    paths = get_paths_by_domain("dss")
    df_path = paths["data_csv_path"]
    df = pd.read_csv(df_path)
    vectorizers = get_bert_models(DOMAIN) + ["trigram", "tfidf", "starr"]
    processor = VectorizerProcessor(
        df, paths["processed_vectorizers_path"], vectorizers
    )
    processed_vectorizers = processor.load_or_generate_embeddings()

    bert_model = "dicta-il/BEREL"
    # model_file = "trained_gae_model_BEREL-finetuned-DSS-maskedLM.pth"
    model_file = "trained_gae_model_BEREL.pth"
    param_dict = {
        "num_adjs": 2,
        "epochs": 50,
        "hidden_dim": 300,
        "distance": "cosine",
        "learning_rate": 0.001,
        "threshold": 0.98,
        "adjacencies": [
            {"type": "tfidf", "params": {"max_features": 10000, "min_df": 0.01}},
            {
                "type": "trigram",
                "params": {
                    "analyzer": "char",
                    "ngram_range": (3, 3),
                    "min_df": 0.01,
                    "max_features": 10000,
                },
            },
        ],
    }
    df_no_nulls = df
    df_no_nulls["section"] = df_no_nulls["section"].fillna("unknown")
    df_no_nulls = df_no_nulls[
        ~(df_no_nulls["composition"].isin(["4QH", "4QM", "4QS", "4QD"]))
    ]
    noisy_compositions = [
        "Para_Gen-Exod",
        "Beatitudes",
    ]
    df_no_nulls = df_no_nulls[~(df_no_nulls["composition"].isin(noisy_compositions))]
    gae_embeddings = get_gae_embeddings(
        df_no_nulls, processed_vectorizers, bert_model, model_file, param_dict
    )
    embeddings_averaged, df_averaged = average_embeddings_by_scroll(
        gae_embeddings, df_no_nulls, "composition", outlier_threshold=2
    )
    df_averaged["composition"] = df_averaged["composition"].apply(format_composition)

    path_to_save = f"{BASE_DIR}/reports/plots/sectarian_dend_GNN_.pdf"
    metrics_gnn = hirerchial_clustering_by_sectarian(
        df_averaged,
        f"Unsupervised Clustering by Sectarian Compositions",
        embeddings_averaged,
        "composition",
        {"linkage_m": "ward", "color_threshold": 0.4},
        path_to_save,
    )
