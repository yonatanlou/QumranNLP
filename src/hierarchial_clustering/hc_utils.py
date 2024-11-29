import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sknetwork.hierarchy import dasgupta_score as calculate_dasgupta_score

from src.baselines.utils import calculate_jaccard_unsupervised, clustering_accuracy
from src.hierarchial_clustering.scrolls_labeling import label_sentence_path


def round_metrics(metrics, n=3):
    new_metrics = {}
    for k, v in metrics.items():
        new_metrics[k] = np.round(v, n)
    return new_metrics


def cluster_and_get_metrics(
    df_labeled_for_clustering, embeddings_tmp, adjacency_matrix_tmp, linkage_m
):
    n_clusters = len(df_labeled_for_clustering["label"].unique())
    linkage_matrix = linkage(embeddings_tmp, method=linkage_m)

    # Calculate Jaccard Index
    flat_clusters = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    le = LabelEncoder()
    true_labels_encoded = le.fit_transform(df_labeled_for_clustering["label"])
    jaccard = calculate_jaccard_unsupervised(true_labels_encoded, flat_clusters)
    silhouette = silhouette_score(embeddings_tmp, flat_clusters)

    dasgupta = calculate_dasgupta_score(adjacency_matrix_tmp, linkage_matrix)
    clustering_acc = clustering_accuracy(true_labels_encoded, flat_clusters)

    metrics = {
        "silhouette": silhouette,
        "jaccard": jaccard,
        "dasgupta": dasgupta,
        "clustering_accuracy": clustering_acc,
    }
    return linkage_matrix, metrics


def generate_dendrogram_plot(
    df_labeled_for_clustering,
    linkage_matrix,
    vec_type,
    metrics,
    curr_scroll,
    label_to_plot="sentence_path",
    path_to_save=None,
):
    if label_to_plot not in ["sentence_path", "label"]:
        raise
    metrics = round_metrics(metrics)

    # Create a color map based on the 'label' column
    unique_labels = df_labeled_for_clustering["label"].unique()
    label_colors = plt.cm.copper(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, label_colors))

    # Plot the dendrogram
    plt.figure(figsize=(10, 14))  # Swap width and height for vertical orientation
    dendrogram(
        linkage_matrix,
        labels=df_labeled_for_clustering[label_to_plot].tolist(),
        orientation="left",  # This makes the dendrogram vertical
        leaf_font_size=6,
        leaf_rotation=0,  # Horizontal text for better readability
    )
    plt.yticks(fontsize=10)

    plt.title(f"embeddings: {vec_type}, {metrics}")
    plt.ylabel("Sentence Path")  # Swap x and y labels
    plt.xlabel("Distance")
    plt.suptitle(f"{curr_scroll} scroll clustering")
    # Color the y-axis labels according to their labels
    ax = plt.gca()
    ylbls = ax.get_ymajorticklabels()
    for idx, lbl in enumerate(ylbls):
        if label_to_plot == "sentence_path":
            label = df_labeled_for_clustering[
                df_labeled_for_clustering["sentence_path"] == lbl.get_text()
            ]["label"].values[0]
        else:
            label = lbl.get_text()
        lbl.set_color(color_map[label])
        lbl.set_text(label)

    # Add a legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markersize=10,
        )
        for label, color in color_map.items()
    ]
    plt.legend(
        handles=legend_elements, title="Labels", loc="best", bbox_to_anchor=(0, 1)
    )

    plt.tight_layout()
    if path_to_save:
        plt.savefig(path_to_save)
    plt.show()


def hirerchial_clustering_by_scroll_gnn(
    df: pd.DataFrame,
    curr_scroll: list[str],
    labels: dict,
    vec_type,
    embeddings,
    adjacency_matrix,
    label_to_plot,
    kwargs,
    path_to_save=None,
):
    linkage_m = kwargs["linkage_m"]

    df = df.reset_index()
    # get labels
    df_labeled_for_clustering = df[df["book"].isin(curr_scroll)]
    df_labeled_for_clustering = label_sentence_path(
        df_labeled_for_clustering, labels, verbose=False
    )
    idxs_global = df_labeled_for_clustering["index"]
    # get embeddings
    label_idxs = list(df_labeled_for_clustering.index)
    adjacency_matrix_tmp = adjacency_matrix[idxs_global, :][:, idxs_global]
    embeddings_tmp = embeddings[label_idxs]
    if type(embeddings_tmp) == scipy.sparse._csr.csr_matrix:
        embeddings_tmp = embeddings_tmp.toarray()

    # get metrics and linkage
    linkage_matrix, metrics = cluster_and_get_metrics(
        df_labeled_for_clustering, embeddings_tmp, adjacency_matrix_tmp, linkage_m
    )

    # plot
    if label_to_plot:
        generate_dendrogram_plot(
            df_labeled_for_clustering,
            linkage_matrix,
            vec_type,
            metrics,
            curr_scroll,
            label_to_plot,
            path_to_save,
        )
    return metrics
