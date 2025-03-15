import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics.pairwise import cosine_similarity

from config import get_paths_by_domain
from src.baselines.embeddings import get_bert_models, VectorizerProcessor
from src.plots_generation.analysis_utils import get_gae_embeddings
from src.website.plotting_utils import plot_embeddings_projection


def filter_data(df):
    df = df.copy()
    df["section"] = df["section"].fillna("Unknown")
    df = df[~df["composition"].isin(["4QH", "4QM", "4QS", "4QD"])]
    noisy_compositions = ["Para_Gen-Exod", "Beatitudes"]
    df = df[~df["composition"].isin(noisy_compositions)]
    return df


def remove_outliers(df, gae_embeddings, outlier_threshold=2):
    # Ensure groupby_col is defined
    groupby_col = "composition"  # Adjust to match the appropriate column name
    unique_books = df.dropna(subset=[groupby_col], axis=0)[groupby_col].unique()

    new_embeddings = []
    new_df = pd.DataFrame()

    for book in unique_books:
        # Get the embeddings for the current book
        book_mask = df[groupby_col] == book
        book_indices = np.where(book_mask)[0]
        book_embeddings = gae_embeddings[book_indices]

        # Compute the initial average embedding
        avg_embedding = np.mean(book_embeddings, axis=0, keepdims=True)

        # Calculate cosine similarities of each embedding to the average
        similarities = cosine_similarity(book_embeddings, avg_embedding).flatten()

        # Compute Z-scores for the cosine similarities
        similarity_z_scores = zscore(similarities)

        # Filter out embeddings with Z-scores above the threshold
        valid_indices = np.abs(similarity_z_scores) <= outlier_threshold
        valid_embeddings = book_embeddings[valid_indices]

        # Append valid embeddings to the list
        new_embeddings.extend(valid_embeddings)

        # Append valid rows to new_df
        valid_df_rows = df.loc[book_mask].iloc[valid_indices]
        new_df = pd.concat([new_df, valid_df_rows], ignore_index=True)

        print(
            "Removed",
            book_indices.shape[0] - valid_embeddings.shape[0],
            "outliers out of",
            book_indices.shape[0],
        )

    # Convert the new embeddings list to a numpy array for consistency
    new_gae_embeddings = np.array(new_embeddings)
    return new_df, new_gae_embeddings


def generate_scatter_plots():
    DOMAIN = "dss"
    paths = get_paths_by_domain(DOMAIN)
    df_path = paths["data_csv_path"]
    df = pd.read_csv(df_path)
    vectorizers = get_bert_models(DOMAIN) + ["trigram", "tfidf"]
    processor = VectorizerProcessor(df, paths["processed_vectorizers_path"], vectorizers)
    processed_vectorizers = processor.load_or_generate_embeddings()

    bert_model = "dicta-il/BEREL"
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

    new_df, new_gae_embeddings = remove_outliers(
        df_no_nulls, gae_embeddings, outlier_threshold=2
    )

    # Create the two figures
    fig1 = plot_embeddings_projection(
        new_gae_embeddings, new_df, color_by="Section", method="tsne", random_state=42
    )
    fig2 = plot_embeddings_projection(
        new_gae_embeddings, new_df, color_by="Composition", method="tsne", random_state=42
    )
    return fig1, fig2
