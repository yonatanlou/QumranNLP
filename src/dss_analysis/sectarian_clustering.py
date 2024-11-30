import numpy as np
import pandas as pd

from config import BASE_DIR
from config import get_paths_by_domain

import scipy
from src.dss_analysis.analysis_utils import (
    get_gvae_embeddings,
    generate_dendrogram_plot,
    cluster_and_get_metrics,
)
from src.baselines.utils import (
    set_seed_globally,
)
from src.baselines.embeddings import VectorizerProcessor, get_bert_models


def hirerchial_clustering_by_sectarian(
    df: pd.DataFrame,
    vec_type,
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
            vec_type,
            metrics,
            "sectarian",
            label_to_plot,
            path_to_save,
        )
    return metrics


def average_embeddings_by_scroll(X, df_no_nulls):
    unique_books = df_no_nulls["book"].unique()
    averaged_embeddings = []
    new_df = pd.DataFrame()

    for book in unique_books:
        book_mask = df_no_nulls["book"] == book
        book_indices = np.where(book_mask)[0]

        book_embeddings = X[book_indices]
        avg_embedding = np.mean(book_embeddings, axis=0)
        averaged_embeddings.append(avg_embedding)

        # Add book info to new DataFrame
        book_df = df_no_nulls[book_mask].head(1).copy()
        new_df = pd.concat([new_df, book_df], ignore_index=True)

    # Convert averaged embeddings to numpy array
    averaged_embeddings = np.array(averaged_embeddings)

    return averaged_embeddings, new_df


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
    model_file = "trained_gvae_model_BEREL-finetuned-DSS-maskedLM.pth"
    param_dict = {
        "num_adjs": 1,
        "distance": "cosine",
        "threshold": 0.97,
        "adjacencies": [
            {"type": "tfidf", "params": {"max_features": 4000, "min_df": 0.001}}
        ],
    }
    df_no_nulls = df[~df["section"].isna()]
    gvae_embeddings = get_gvae_embeddings(
        df_no_nulls, processed_vectorizers, bert_model, model_file, param_dict
    )
    embeddings_averaged, df_averaged = average_embeddings_by_scroll(
        gvae_embeddings, df_no_nulls
    )

    path_to_save = f"{BASE_DIR}/reports/plots/sectarian_dend_GNN.png"
    metrics_gnn = hirerchial_clustering_by_sectarian(
        df_averaged,
        f"GNN with {bert_model} embeddings + TF-IDF edges",
        embeddings_averaged,
        "book",
        {"linkage_m": "ward"},
        path_to_save,
    )
