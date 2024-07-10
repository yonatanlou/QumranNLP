from datetime import datetime

import pandas as pd
from tqdm.notebook import tqdm

from notebooks.features import create_adjacency_matrix, vectorize_text
from notebooks.clustering import (
    get_random_clusters_score,
    get_clusters_scores,
    get_dendrogram_feature_order,
    get_metrics_from_LCA,
)


def run_clustering_cv(
    processed_vectorizers,
    df,
    frac,
    num_cvs,
    context_similiarity_window,
    vectorizers,
    linkage_method,
):
    scores = []
    for i in tqdm(range(num_cvs)):
        # Sample 90% of the data
        sampled_df = stratified_sample(df, "book", frac=frac, random_state=42 + i)
        idxs = sampled_df["original_index"]
        print(f"{datetime.now()} - {sampled_df.shape=}")
        adjacency_matrix = create_adjacency_matrix(
            sampled_df,
            context_similiarity_window=context_similiarity_window,
            composition_level=True,
        )

        for vectorizer_type in tqdm(vectorizers):
            # vectorizer_matrix = vectorize_text(sampled_df, "text", vectorizer_type)
            vectorizer_matrix = processed_vectorizers[vectorizer_type]
            vectorizer_matrix = vectorizer_matrix[idxs]
            print(f"{datetime.now()} - {vectorizer_type=},{vectorizer_matrix.shape=}")

            dasgupta_score, linkage_matrix = get_clusters_scores(
                vectorizer_matrix, linkage_method, adjacency_matrix
            )
            dasgupta_score_rand = get_random_clusters_score(
                sampled_df, "sentence_path", vectorizer_matrix, "ward", iters=2
            )
            print(f"{dasgupta_score=}, {dasgupta_score_rand=}\n")

            feature_order = get_dendrogram_feature_order(
                linkage_matrix, df["sentence_path"].to_list()
            )
            _, LCA_metric_mean, LCA_metric_std = get_metrics_from_LCA(feature_order)

            scores.append(
                {
                    "vectorizer": vectorizer_type,
                    "dasgupta_score": dasgupta_score,
                    "dasgupta_score_rand": dasgupta_score_rand,
                    "max_dist_metric_mean": LCA_metric_mean,
                    "cv": i,
                }
            )

    # Convert scores to a DataFrame for easier analysis
    scores_df = pd.DataFrame(scores)
    return scores_df


def stratified_sample(df, stratify_column, frac, random_state=None):
    df = df.reset_index().rename(columns={"index": "original_index"})
    grouped = df.groupby(stratify_column)
    stratified_df = grouped.apply(
        lambda x: x.sample(frac=frac, random_state=random_state)
    ).reset_index(drop=True)
    stratified_df = stratified_df.sort_values(by="original_index")
    return stratified_df
