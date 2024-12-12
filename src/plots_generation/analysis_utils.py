import os

import numpy as np
import pandas as pd
import scipy
import torch
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sknetwork.hierarchy import dasgupta_score as calculate_dasgupta_score

from config import BASE_DIR
from src.baselines.utils import calculate_jaccard_unsupervised, clustering_accuracy
from src.plots_generation.scrolls_labeling import label_sentence_path
from src.gnn.adjacency import AdjacencyMatrixGenerator
from src.gnn.model import GVAE


def round_metrics(metrics, n=3):
    new_metrics = {}
    for k, v in metrics.items():
        new_metrics[k] = np.round(v, n)
    return new_metrics


def cluster_and_get_metrics(
    df_labeled_for_clustering, embeddings_tmp, adjacency_matrix_tmp, linkage_m
):
    n_clusters = len(df_labeled_for_clustering["label"].unique())
    if linkage_m in ["centroid", "ward", "median"]:
        clust_metric = "euclidean"
    else:
        clust_metric = "cosine"
    linkage_matrix = linkage(embeddings_tmp, method=linkage_m, metric=clust_metric)

    # Calculate Jaccard Index
    flat_clusters = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    le = LabelEncoder()
    true_labels_encoded = le.fit_transform(df_labeled_for_clustering["label"])
    jaccard = calculate_jaccard_unsupervised(true_labels_encoded, flat_clusters)
    silhouette = silhouette_score(embeddings_tmp, flat_clusters)
    if adjacency_matrix_tmp is not None:
        dasgupta = calculate_dasgupta_score(adjacency_matrix_tmp, linkage_matrix)
    else:
        dasgupta = None
    clustering_acc = clustering_accuracy(true_labels_encoded, flat_clusters)

    metrics = {
        "silhouette": silhouette,
        "jaccard": jaccard,
        "clustering_accuracy": clustering_acc,
    }
    if dasgupta:
        metrics["dasgupta"] = dasgupta
    return linkage_matrix, metrics


def generate_dendrogram_plot(
    df_labeled_for_clustering,
    linkage_matrix,
    title,
    metrics,
    label_to_plot="sentence_path",
    path_to_save=None,
    kwargs={},
):
    import scienceplots

    from matplotlib.lines import Line2D

    # Font sizes and styles for publication
    XTICK_FONT_SIZE = 17
    XLAB_FONT_SIZE = 19
    TITLE_FONT_SIZE = 24
    LEGEND_FONT_SIZE = 16

    # sectarian
    # XTICK_FONT_SIZE = 20
    # XLAB_FONT_SIZE = 23
    # TITLE_FONT_SIZE = 38
    # LEGEND_FONT_SIZE = 20

    plt.style.use(["science"])  # Ensure science plots with no LaTeX errors

    if label_to_plot not in ["sentence_path", "label", "book", "composition"]:
        raise ValueError("Invalid label_to_plot value!")

    # Format metrics
    # metrics = round_metrics(metrics)
    df_labeled_for_clustering["composition"] = df_labeled_for_clustering[
        "composition"
    ].fillna(df_labeled_for_clustering["book"])

    # Create a color map based on the 'label' column
    unique_labels = df_labeled_for_clustering["label"].unique()
    label_colors = plt.cm.copper(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, label_colors))

    # Plot the dendrogram
    fig, ax = plt.subplots(figsize=(7, 18.5), dpi=100)
    # fig, ax = plt.subplots(figsize=(7, 16), dpi=100) #sectarian
    color_thres = kwargs.get("color_threshold", 0.7)
    dendrogram(
        linkage_matrix,
        labels=df_labeled_for_clustering[label_to_plot].tolist(),
        orientation="left",  # Vertical dendrogram
        leaf_font_size=XTICK_FONT_SIZE,
        leaf_rotation=0,  # Horizontal labels
        ax=ax,
        color_threshold=color_thres * max(linkage_matrix[:, 2]),
    )
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set_ylabel(label_to_plot.capitalize(), fontsize=XLAB_FONT_SIZE)
    ax.set_xlabel("Distance", fontsize=XLAB_FONT_SIZE)

    # Customize y-axis labels with colors
    ylbls = ax.get_ymajorticklabels()
    for lbl in ylbls:
        text = lbl.get_text()
        label = df_labeled_for_clustering[
            df_labeled_for_clustering[label_to_plot] == text
        ]["label"].values[0]
        lbl.set_color(color_map[label])
        lbl.set_text(label)

    # Add a legend at the bottom
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markersize=8,
        )
        for label, color in color_map.items()
    ]
    fig.legend(
        handles=legend_elements,
        title="Labels",
        loc="lower center",
        ncol=2,  # Two-column legend
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
        bbox_to_anchor=(0.5, 0),  # Position at the bottom
        frameon=True,
    )
    plt.subplots_adjust(bottom=0.1, top=0.99)
    # sectarian
    # plt.subplots_adjust(
    #     bottom=0.13, top=0.99
    # )

    # plt.tight_layout()
    # Save or display the plot
    if path_to_save:
        plt.savefig(path_to_save, bbox_inches="tight")
        print(f"Saved plot to {path_to_save}")
    # plt.show()


def hirerchial_clustering_by_scroll_gnn(
    df: pd.DataFrame,
    curr_scroll: list[str],
    labels: dict,
    title,
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
    if isinstance(embeddings_tmp, scipy.sparse.csr_matrix):
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
            title,
            metrics,
            label_to_plot,
            path_to_save,
        )
    return metrics


def get_gvae_embeddings(df, processed_vectorizers, bert_model, model_file, param_dict):
    param_dict["bert_model"] = bert_model

    # Predefined paths and configurations
    models_dir = f"{BASE_DIR}/models"

    # Generate adjacency matrix
    adj_gen = AdjacencyMatrixGenerator(
        vectorizer_type=param_dict["adjacencies"][0]["type"],
        vectorizer_params=param_dict["adjacencies"][0]["params"],
        threshold=param_dict["threshold"],
        distance_metric=param_dict["distance"],
        meta_params=None,
        normalize=True,
    )

    # Generate graph
    edge_index, edge_attr, adj_matrix = adj_gen.generate_graph(df)

    # Prepare model
    model_path = os.path.join(models_dir, model_file)
    kwargs, state = torch.load(model_path)

    model = GVAE(**kwargs)
    model.load_state_dict(state)
    model.eval()

    # Prepare input
    X = processed_vectorizers[bert_model]
    X = X[df.index]
    X = X.astype("float32")
    X_tensor = torch.FloatTensor(X)

    # Get embeddings
    with torch.no_grad():
        _, mu, *_ = model(X_tensor, edge_index, edge_attr)
    embeddings_gvae = mu.numpy()
    return embeddings_gvae
