from collections import Counter
from datetime import datetime
from itertools import groupby

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.cluster.hierarchy import dendrogram
from sklearn import cluster as sk
from sknetwork.hierarchy import dasgupta_score as calculate_dasgupta_score
from tqdm.notebook import tqdm

from src.baselines.features import get_linkage_matrix


def get_metrics_from_LCA(feature_order):
    n_chunks_df = pd.DataFrame(
        [Counter([i.split(":")[0] for i in feature_order])], index=["n_chunks"]
    ).T
    scrolls = [item.split(":")[0] for item in feature_order]

    # Initialize a dictionary to store the results
    scroll_analysis = {}

    # Group by consecutive scrolls
    for scroll, group in groupby(scrolls):
        group_list = list(group)
        if scroll not in scroll_analysis:
            scroll_analysis[scroll] = {
                "sequences": [],
            }
        scroll_analysis[scroll]["sequences"].append(len(group_list))

    # Compute the number of sequences and mean sequence length for each scroll
    for scroll in scroll_analysis:
        sequences = scroll_analysis[scroll]["sequences"]
        number_of_sequences = len(sequences)
        scroll_analysis[scroll] = {
            "n_sequences": number_of_sequences,
        }
    res = pd.DataFrame(scroll_analysis).T.join(n_chunks_df)
    res["1-n_sequences/n_chunks"] = 1 - res["n_sequences"] / res["n_chunks"]

    # Initialize a dictionary to store the first and last occurrence of each scroll
    scroll_positions = {}

    for index, scroll in enumerate(scrolls):
        if scroll not in scroll_positions:
            scroll_positions[scroll] = [
                index,
                index,
            ]  # [first_occurrence, last_occurrence]
        else:
            scroll_positions[scroll][1] = index  # Update last_occurrence

    # Calculate the maximum distance between the first and last occurrence for each scroll
    max_distances = {
        scroll: positions[1] - positions[0] + 1
        for scroll, positions in scroll_positions.items()
    }
    max_distances = (
        pd.DataFrame([max_distances])
        .T.rename(columns={0: "max_dist"})
        .sort_values(by="max_dist")
    )
    final_res = max_distances.join(res)
    final_res["n_chunks/max_dist"] = final_res["n_chunks"] / final_res["max_dist"]
    return (
        final_res.sort_values(by="n_chunks/max_dist"),
        final_res["n_chunks/max_dist"].mean(),
        final_res["n_chunks/max_dist"].std(),
    )


def get_dendrogram_feature_order(linkage_matrix, sample_names):
    dendro = dendrogram(
        linkage_matrix,
        leaf_label_func=lambda x: sample_names[x],
        orientation="right",
        no_plot=True,  # Do not plot the dendrogram
    )

    feature_order = [sample_names[i] for i in dendro["leaves"]]
    return feature_order


def get_clusters_scores(vectorizer_matrix, linkage_criterion, adjacency_matrix):
    if sp.issparse(vectorizer_matrix):
        vectorizer_matrix = vectorizer_matrix.toarray()
    model = sk.AgglomerativeClustering(
        distance_threshold=0, n_clusters=None, linkage=linkage_criterion
    )
    print(f"{datetime.now()} - fitting AgglomerativeClustering")
    model.fit_predict(vectorizer_matrix)
    # adjacency_matrix = np.zeros((len(sample_names), len(sample_names)))
    # for i in range(0, adjacency_matrix.shape[0] - 1):
    #     adjacency_matrix[i, i + 1] = 1
    #     adjacency_matrix[i + 1, i] = 1
    print(f"{datetime.now()} - getting linkage matrix")
    linkage_matrix = get_linkage_matrix(model)

    print(f"{datetime.now()} - calculate_dasgupta_score")
    score = calculate_dasgupta_score(adjacency_matrix, linkage_matrix)

    return score, linkage_matrix


def get_random_clusters_score(
    df, label_name, vectorizer_matrix, linkage_criterion, iters=10
):
    random_scores = []
    if sp.issparse(vectorizer_matrix):
        vectorizer_matrix = vectorizer_matrix.toarray()
    sample_names = df[label_name].to_list()
    print(f"{datetime.now()} - calculate rand dasgupta_score")
    for i in tqdm(range(iters)):
        indexes = np.arange(len(vectorizer_matrix))
        np.random.shuffle(indexes)
        model = sk.AgglomerativeClustering(
            distance_threshold=0, n_clusters=None, linkage="ward"
        )
        model.fit_predict(vectorizer_matrix[indexes])
        linkage_matrix = get_linkage_matrix(model)
        adjacency_matrix = np.zeros((len(sample_names), len(sample_names)))
        for i in range(0, adjacency_matrix.shape[0] - 1):
            adjacency_matrix[i, i + 1] = 1
            adjacency_matrix[i + 1, i] = 1

        dasgupta_score_rand = calculate_dasgupta_score(adjacency_matrix, linkage_matrix)
        random_scores.append(dasgupta_score_rand)

    return np.mean(np.array(random_scores))
