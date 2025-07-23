import os
import pandas as pd
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.offline as pyo

from config import BASE_DIR, get_paths_by_domain
from src.baselines.embeddings import VectorizerProcessor, get_bert_models
from src.baselines.utils import create_adjacency_matrix, set_seed_globally
from src.gnn.adjacency import AdjacencyMatrixGenerator
from src.plots_generation.analysis_utils import (
    get_gae_embeddings,
    cluster_and_get_metrics,
)
from src.plots_generation.scrolls_labeling import (
    labels_1QS,
    labels_hodayot,
    labels_1QM,
    label_sentence_path,
)

exp_dir = f"{BASE_DIR}/experiments/dss/clustering_by_scroll"
DOMAIN = "dss"
paths = get_paths_by_domain(DOMAIN)
DATA_CSV_PATH = paths["data_csv_path"]

df = pd.read_csv(DATA_CSV_PATH)

vectorizers = (get_bert_models(DOMAIN) or []) + ["trigram", "tfidf", "starr"]
processor = VectorizerProcessor(df, paths["processed_vectorizers_path"], vectorizers)
PROCESSED_VECTORIZERS = processor.load_or_generate_embeddings()
ADJACENCY_MATRIX_ALL = create_adjacency_matrix(
    df,
    context_similiarity_window=3,
    composition_level=False,
)


def generate_2d_embeddings(embeddings, method="tsne", random_state=42):
    """Generate 2D embeddings using t-SNE or PCA"""
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=min(30, embeddings.shape[0]-1))
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    else:
        raise ValueError("Method should be either 'tsne' or 'pca'")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    return embeddings_2d


def create_scatter_plot(df_labeled, embeddings_2d, title, method_name, scroll_name):
    """Create a plotly scatter plot for the clustering visualization"""
    
    # Create unique colors for each label
    unique_labels = df_labeled["label"].unique()
    colors = px.colors.qualitative.Set3[:len(unique_labels)]
    color_map = dict(zip(unique_labels, colors))
    
    # Create the scatter plot
    fig = go.Figure()
    
    for label in unique_labels:
        mask = df_labeled["label"] == label
        embeddings_subset = embeddings_2d[mask]
        
        fig.add_trace(go.Scatter(
            x=embeddings_subset[:, 0],
            y=embeddings_subset[:, 1],
            mode='markers',
            name=label,
            marker=dict(
                color=color_map[label],
                size=8,
                line=dict(width=1, color='black')
            ),
            text=df_labeled[mask]["sentence_path"].tolist(),
            hovertemplate='<b>%{text}</b><br>Label: ' + label + '<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"{title}<br><sub>{method_name} - {scroll_name}</sub>",
        xaxis_title=f"Component 1",
        yaxis_title=f"Component 2",
        font=dict(size=12),
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    return fig


def scatter_clustering_by_scroll_gnn(
    df: pd.DataFrame,
    curr_scroll: list[str],
    labels: dict,
    title: str,
    embeddings,
    adjacency_matrix,
    label_to_plot: str,
    reduction_method: str = "tsne",
    kwargs=None,
):
    """Generate scatter plot clustering visualization similar to hierarchical clustering function"""
    
    if kwargs is None:
        kwargs = {"linkage_m": "ward"}
    
    linkage_m = kwargs["linkage_m"]
    
    df = df.reset_index()
    # Get labels
    df_labeled_for_clustering = df[df["book"].isin(curr_scroll)]
    df_labeled_for_clustering = label_sentence_path(
        df_labeled_for_clustering, labels, verbose=False
    )
    idxs_global = df_labeled_for_clustering["index"]
    
    # Get embeddings
    label_idxs = list(df_labeled_for_clustering.index)
    adjacency_matrix_tmp = adjacency_matrix[idxs_global, :][:, idxs_global]
    embeddings_tmp = embeddings[label_idxs]
    
    if hasattr(embeddings_tmp, 'toarray'):  # Check if sparse matrix
        embeddings_tmp = embeddings_tmp.toarray()
    
    # Get metrics (same as original for consistency)
    linkage_matrix, metrics = cluster_and_get_metrics(
        df_labeled_for_clustering, embeddings_tmp, adjacency_matrix_tmp, linkage_m
    )
    
    # Generate 2D embeddings for visualization
    embeddings_2d = generate_2d_embeddings(embeddings_tmp, method=reduction_method)
    
    # Create scatter plot
    scroll_str = str(curr_scroll)
    fig = create_scatter_plot(
        df_labeled_for_clustering, 
        embeddings_2d, 
        title, 
        reduction_method.upper(),
        scroll_str
    )
    
    return fig, metrics


def get_scatter_plots_per_model(
    bert_model: str, 
    model_file: str, 
    param_dict: dict, 
    label_to_plot: str,
    reduction_method: str = "tsne"
) -> tuple[list, list]:
    """Generate scatter plots for GNN model embeddings"""
    
    set_seed_globally()
    models_dir = f"{BASE_DIR}/models"
    data_csv_path = paths["data_csv_path"]
    param_dict["bert_model"] = bert_model

    # Read and filter DataFrame
    df = pd.read_csv(data_csv_path)
    df_ = df[df["book"].isin(["1QM", "1QSa", "1QS", "1QHa"])]
    embeddings_gvae = get_gae_embeddings(
        df_, PROCESSED_VECTORIZERS, bert_model, model_file, param_dict
    )
    
    # Labels setup
    labels_all = {
        "['1QHa']": labels_hodayot,
    }

    all_results = []
    all_plots = []
    
    for scroll, labels in labels_all.items():
        fig, metrics_gnn = scatter_clustering_by_scroll_gnn(
            df_,
            eval(scroll),
            labels,
            f"Unsupervised Clustering of the Hodayot Composition",
            embeddings_gvae,
            ADJACENCY_MATRIX_ALL,
            label_to_plot,
            reduction_method,
        )

        # Add metadata to results
        metrics_gnn.update(
            {
                "scroll": scroll,
                "vectorizer": bert_model,
                "model_file": model_file,
                "method": "GNN",
            }
        )

        all_results.append(metrics_gnn)
        all_plots.append(fig)

    return all_plots, all_results


def get_scatter_plots_per_vectorizer(
    vectorizer_name: str, 
    label_to_plot: str,
    reduction_method: str = "tsne"
) -> tuple[list, list]:
    """Generate scatter plots for direct vectorizer embeddings"""
    
    set_seed_globally()
    data_csv_path = paths["data_csv_path"]
    
    # Read and filter DataFrame
    df = pd.read_csv(data_csv_path)
    df_ = df[df["book"].isin(["1QM", "1QSa", "1QS", "1QHa"])].copy()
    
    # Get embeddings from processor
    embeddings = PROCESSED_VECTORIZERS[vectorizer_name]
    embeddings = embeddings[df_.index]
    
    # Labels setup
    labels_all = {
        "['1QHa']": labels_hodayot,
    }

    all_results = []
    all_plots = []
    
    for scroll, labels in labels_all.items():
        # Create title based on vectorizer method
        method_title = f"Unsupervised Clustering using {vectorizer_name.upper()} Embeddings"
        
        fig, metrics_vectorizer = scatter_clustering_by_scroll_gnn(
            df_,
            eval(scroll),
            labels,
            method_title,
            embeddings,
            ADJACENCY_MATRIX_ALL,
            label_to_plot,
            reduction_method,
        )

        # Add metadata to results
        metrics_vectorizer.update(
            {
                "scroll": scroll,
                "vectorizer": vectorizer_name,
                "method": "direct_embedding",
            }
        )

        all_results.append(metrics_vectorizer)
        all_plots.append(fig)

    return all_plots, all_results


def save_combined_html(plots_list, output_path):
    """Save all plots to a single HTML file"""
    
    # Create subplots layout
    n_plots = len(plots_list)
    cols = 2  # Two columns
    rows = (n_plots + cols - 1) // cols  # Ceiling division
    
    # Create subplot titles
    subplot_titles = []
    for i, plot in enumerate(plots_list):
        title = plot.layout.title.text if plot.layout.title else f"Plot {i+1}"
        subplot_titles.append(title)
    
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
        specs=[[{"type": "scatter"}] * cols for _ in range(rows)]
    )
    
    # Add traces from each plot
    for idx, plot in enumerate(plots_list):
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        
        for trace in plot.data:
            fig.add_trace(
                go.Scatter(
                    x=trace.x,
                    y=trace.y,
                    mode=trace.mode,
                    name=f"{trace.name}",
                    marker=trace.marker,
                    text=trace.text,
                    hovertemplate=trace.hovertemplate,
                    showlegend=True if idx == 0 else False  # Only show legend for first plot
                ),
                row=row, 
                col=col
            )
    
    fig.update_layout(
        height=400 * rows,
        width=1600,
        title_text="Clustering Within Scroll - Scatter Plot Analysis",
        title_x=0.5,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # Save to HTML
    pyo.plot(fig, filename=output_path, auto_open=False)
    print(f"Combined scatter plots saved to: {output_path}")


if __name__ == "__main__":
    # Best parameters by Hodayot
    param_dict = {
        "num_adjs": 2,
        "epochs": 50,
        "hidden_dim": 300,
        "distance": "cosine",
        "learning_rate": 0.001,
        "threshold": 0.9,
        "adjacencies": [
            {"type": "tfidf", "params": {"max_features": 1000, "min_df": 0.001}},
            {
                "type": "trigram",
                "params": {
                    "analyzer": "char",
                    "ngram_range": (3, 3),
                    "min_df": 0.001,
                    "max_features": 1000,
                },
            },
        ],
    }
    bert_model = "dicta-il/BEREL"
    model_file = "trained_gae_model_BEREL.pth"
    output_dir = f"{BASE_DIR}/reports/plots"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    all_plots = []
    all_results = []
    
    print("Generating scatter plots with GAE embeddings...")
    gae_plots, gae_results = get_scatter_plots_per_model(
        bert_model,
        model_file,
        param_dict,
        label_to_plot="sentence_path",
        reduction_method="tsne"
    )
    all_plots.extend(gae_plots)
    all_results.extend(gae_results)
    print(f"GAE results: {gae_results}")
    
    print("Generating scatter plots with TF-IDF embeddings...")
    tfidf_plots, tfidf_results = get_scatter_plots_per_vectorizer(
        "tfidf",
        label_to_plot="sentence_path",
        reduction_method="tsne"
    )
    all_plots.extend(tfidf_plots)
    all_results.extend(tfidf_results)
    print(f"TF-IDF results: {tfidf_results}")

    print("Generating scatter plots with trigram embeddings...")
    trigram_plots, trigram_results = get_scatter_plots_per_vectorizer(
        "trigram",
        label_to_plot="sentence_path",
        reduction_method="tsne"
    )
    all_plots.extend(trigram_plots)
    all_results.extend(trigram_results)
    print(f"Trigram results: {trigram_results}")

    # Save combined HTML
    output_html_path = os.path.join(output_dir, "clustering_within_scroll_scatter_plots.html")
    save_combined_html(all_plots, output_html_path)
    
    print("Scatter plot generation completed!")