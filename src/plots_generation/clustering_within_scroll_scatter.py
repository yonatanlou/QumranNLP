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


def generate_3d_embeddings(embeddings, method="tsne", random_state=42):
    """Generate 3D embeddings using t-SNE or PCA"""
    if method == "tsne":
        reducer = TSNE(n_components=3, random_state=random_state, perplexity=min(30, embeddings.shape[0]-1))
    elif method == "pca":
        reducer = PCA(n_components=3, random_state=random_state)
    else:
        raise ValueError("Method should be either 'tsne' or 'pca'")
    
    embeddings_3d = reducer.fit_transform(embeddings)
    return embeddings_3d


def create_scatter_plot(df_labeled, embeddings, title, method_name, scroll_name, plot_id, is_3d=False):
    """Create a plotly scatter plot for the clustering visualization"""
    
    # Create unique colors for each label
    unique_labels = df_labeled["label"].unique()
    colors = px.colors.qualitative.Set3[:len(unique_labels)]
    color_map = dict(zip(unique_labels, colors))
    
    # Create the scatter plot
    fig = go.Figure()
    
    for label in unique_labels:
        mask = df_labeled["label"] == label
        embeddings_subset = embeddings[mask]
        sentence_paths = df_labeled[mask]["sentence_path"].tolist()
        
        if is_3d:
            # Add normal trace for each label (3D)
            fig.add_trace(go.Scatter3d(
                x=embeddings_subset[:, 0],
                y=embeddings_subset[:, 1],
                z=embeddings_subset[:, 2],
                mode='markers',
                name=label,
                marker=dict(
                    color=color_map[label],
                    size=5,
                    line=dict(width=1, color='black')
                ),
                text=sentence_paths,
                customdata=sentence_paths,  # Store sentence paths for JavaScript access
                hovertemplate='<b>%{text}</b><br>Label: ' + label + '<extra></extra>',
                visible=True
            ))
            
            # Add highlight trace for each label (initially invisible, 3D)
            fig.add_trace(go.Scatter3d(
                x=embeddings_subset[:, 0],
                y=embeddings_subset[:, 1],
                z=embeddings_subset[:, 2],
                mode='markers',
                name=f'{label}_highlight',
                marker=dict(
                    color='red',
                    size=8,
                    line=dict(width=2, color='darkred'),
                    symbol='diamond'
                ),
                text=sentence_paths,
                customdata=sentence_paths,
                hovertemplate='<b>%{text}</b><br>Label: ' + label + ' (SELECTED)<extra></extra>',
                visible=False,
                showlegend=False
            ))
        else:
            # Add normal trace for each label (2D)
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
                text=sentence_paths,
                customdata=sentence_paths,  # Store sentence paths for JavaScript access
                hovertemplate='<b>%{text}</b><br>Label: ' + label + '<extra></extra>',
                visible=True
            ))
            
            # Add highlight trace for each label (initially invisible, 2D)
            fig.add_trace(go.Scatter(
                x=embeddings_subset[:, 0],
                y=embeddings_subset[:, 1],
                mode='markers',
                name=f'{label}_highlight',
                marker=dict(
                    color='red',
                    size=12,
                    line=dict(width=2, color='darkred'),
                    symbol='star'
                ),
                text=sentence_paths,
                customdata=sentence_paths,
                hovertemplate='<b>%{text}</b><br>Label: ' + label + ' (SELECTED)<extra></extra>',
                visible=False,
                showlegend=False
            ))
    
    if is_3d:
        fig.update_layout(
            title=f"{title}<br><sub>{method_name} 3D - {scroll_name}</sub>",
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3"
            ),
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
    else:
        fig.update_layout(
            title=f"{title}<br><sub>{method_name} 2D - {scroll_name}</sub>",
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
    is_3d: bool = False,
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
    
    # Generate embeddings for visualization based on dimensionality
    if is_3d:
        embeddings_vis = generate_3d_embeddings(embeddings_tmp, method=reduction_method)
    else:
        embeddings_vis = generate_2d_embeddings(embeddings_tmp, method=reduction_method)
    
    # Create scatter plot
    scroll_str = str(curr_scroll)
    plot_id = f"{reduction_method}_{hash(title) % 10000}"  # Create unique plot ID
    fig = create_scatter_plot(
        df_labeled_for_clustering, 
        embeddings_vis, 
        title, 
        reduction_method.upper(),
        scroll_str,
        plot_id,
        is_3d=is_3d
    )
    
    return fig, metrics, df_labeled_for_clustering


def get_scatter_plots_per_model(
    bert_model: str, 
    model_file: str, 
    param_dict: dict, 
    label_to_plot: str,
    reduction_method: str = "tsne",
    is_3d: bool = False
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
    all_dataframes = []
    
    for scroll, labels in labels_all.items():
        fig, metrics_gnn, df_labeled = scatter_clustering_by_scroll_gnn(
            df_,
            eval(scroll),
            labels,
            f"Unsupervised Clustering of the Using GNN Embeddings (BEREL based)",
            embeddings_gvae,
            ADJACENCY_MATRIX_ALL,
            label_to_plot,
            reduction_method,
            is_3d,
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
        all_dataframes.append(df_labeled)

    return all_plots, all_results, all_dataframes


def get_scatter_plots_per_vectorizer(
    vectorizer_name: str, 
    label_to_plot: str,
    reduction_method: str = "tsne",
    is_3d: bool = False
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
    all_dataframes = []
    
    for scroll, labels in labels_all.items():
        # Create title based on vectorizer method
        method_title = f"Unsupervised Clustering using {vectorizer_name.upper()} Embeddings"
        
        fig, metrics_vectorizer, df_labeled = scatter_clustering_by_scroll_gnn(
            df_,
            eval(scroll),
            labels,
            method_title,
            embeddings,
            ADJACENCY_MATRIX_ALL,
            label_to_plot,
            reduction_method,
            is_3d,
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
        all_dataframes.append(df_labeled)

    return all_plots, all_results, all_dataframes


def save_combined_html(plots_list, dataframes_list, output_path):
    """Save all plots to a single interactive HTML file with sentence path selector"""
    
    # Collect all unique sentence paths from all dataframes
    all_sentence_paths = set()
    for df in dataframes_list:
        all_sentence_paths.update(df["sentence_path"].tolist())
    all_sentence_paths = sorted(list(all_sentence_paths))
    
    # Convert plots to JSON for JavaScript
    import json
    import numpy as np
    
    def convert_numpy_to_list(obj):
        """Recursively convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_list(item) for item in obj]
        else:
            return obj
    
    plots_json = []
    for i, plot in enumerate(plots_list):
        plot_dict = plot.to_dict()
        # Convert numpy arrays to lists for JSON serialization
        plot_dict_serializable = convert_numpy_to_list(plot_dict)
        plots_json.append({
            'plot_id': f'plot_{i}',
            'data': plot_dict_serializable['data'],
            'layout': plot_dict_serializable['layout']
        })
    
    # Create the HTML template with JavaScript interaction
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Clustering Within Scroll Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .container {{
            display: flex;
            gap: 20px;
        }}
        .sidebar {{
            width: 300px;
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 5px;
            height: fit-content;
        }}
        .plots-container {{
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .plot-div {{
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        select {{
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
            border: 1px solid #ccc;
            border-radius: 3px;
        }}
        .info {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        button {{
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            width: 100%;
        }}
        button:hover {{
            background-color: #0056b3;
        }}
        .selected-info {{
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 3px;
            margin-top: 10px;
            border: 1px solid #ffeaa7;
        }}
    </style>
</head>
<body>
    <h1>Interactive Clustering Within Scroll Analysis</h1>
    
    <div class="container">
        <div class="sidebar">
            <div class="info">
                <h3>Sentence Path Selector</h3>
                <p>Select a sentence path to highlight the corresponding points across all plots in red stars.</p>
            </div>
            
            <label for="sentenceSelect"><strong>Choose Sentence Path:</strong></label>
            <select id="sentenceSelect">
                <option value="">-- Select sentence path --</option>
                {''.join([f'<option value="{path}">{path}</option>' for path in all_sentence_paths])}
            </select>
            
            <button onclick="clearSelection()">Clear Selection</button>
            
            <div id="selectedInfo" class="selected-info" style="display: none;">
                <strong>Selected:</strong> <span id="selectedPath"></span>
            </div>
        </div>
        
        <div class="plots-container">
            {''.join([f'<div id="plot_{i}" class="plot-div"></div>' for i in range(len(plots_list))])}
        </div>
    </div>

    <script>
        // Store plot data
        const plotsData = {json.dumps(plots_json)};
        let currentSelection = null;
        
        // Initialize all plots
        function initializePlots() {{
            plotsData.forEach((plotData, index) => {{
                Plotly.newPlot(`plot_${{index}}`, plotData.data, plotData.layout, {{responsive: true}});
            }});
        }}
        
        // Highlight selected sentence path
        function highlightSentencePath(selectedPath) {{
            if (!selectedPath) {{
                clearSelection();
                return;
            }}
            
            currentSelection = selectedPath;
            document.getElementById('selectedPath').textContent = selectedPath;
            document.getElementById('selectedInfo').style.display = 'block';
            
            console.log('Highlighting path:', selectedPath);
            
            plotsData.forEach((plotData, plotIndex) => {{
                const plotDiv = `plot_${{plotIndex}}`;
                
                // Get current plot element and data
                const plotElement = document.getElementById(plotDiv);
                if (!plotElement || !plotElement.data) {{
                    console.error('Plot element not found:', plotDiv);
                    return;
                }}
                
                const currentData = plotElement.data;
                console.log(`Plot ${{plotIndex}} has ${{currentData.length}} traces`);
                
                // Process each trace
                for (let traceIndex = 0; traceIndex < currentData.length; traceIndex++) {{
                    const trace = currentData[traceIndex];
                    
                    if (trace.name && trace.name.includes('_highlight')) {{
                        console.log(`Processing highlight trace: ${{trace.name}}`);
                        
                        // Find matching points in the corresponding normal trace
                        const normalTraceIndex = traceIndex - 1; // Highlight traces come after normal traces
                        if (normalTraceIndex >= 0 && normalTraceIndex < currentData.length) {{
                            const normalTrace = currentData[normalTraceIndex];
                            
                            if (normalTrace.customdata) {{
                                const matchingIndices = [];
                                normalTrace.customdata.forEach((path, idx) => {{
                                    if (path === selectedPath) {{
                                        matchingIndices.push(idx);
                                    }}
                                }});
                                
                                console.log(`Found ${{matchingIndices.length}} matching points for ${{selectedPath}}`);
                                
                                if (matchingIndices.length > 0) {{
                                    // Extract data for matching points
                                    const highlightX = matchingIndices.map(idx => normalTrace.x[idx]);
                                    const highlightY = matchingIndices.map(idx => normalTrace.y[idx]);
                                    const highlightText = matchingIndices.map(idx => normalTrace.text[idx]);
                                    const highlightCustomData = matchingIndices.map(idx => normalTrace.customdata[idx]);
                                    
                                    // Check if this is a 3D plot (has z data)
                                    let update;
                                    if (normalTrace.z && normalTrace.z.length > 0) {{
                                        // 3D plot
                                        const highlightZ = matchingIndices.map(idx => normalTrace.z[idx]);
                                        update = {{
                                            x: [highlightX],
                                            y: [highlightY],
                                            z: [highlightZ],
                                            text: [highlightText],
                                            customdata: [highlightCustomData],
                                            visible: [true]
                                        }};
                                    }} else {{
                                        // 2D plot
                                        update = {{
                                            x: [highlightX],
                                            y: [highlightY],
                                            text: [highlightText],
                                            customdata: [highlightCustomData],
                                            visible: [true]
                                        }};
                                    }}
                                    
                                    Plotly.restyle(plotDiv, update, [traceIndex]);
                                }} else {{
                                    // Hide this highlight trace if no matches
                                    Plotly.restyle(plotDiv, {{visible: [false]}}, [traceIndex]);
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        // Clear selection
        function clearSelection() {{
            currentSelection = null;
            document.getElementById('sentenceSelect').value = '';
            document.getElementById('selectedInfo').style.display = 'none';
            
            console.log('Clearing selection');
            
            // Hide all highlight traces
            plotsData.forEach((plotData, plotIndex) => {{
                const plotDiv = `plot_${{plotIndex}}`;
                const plotElement = document.getElementById(plotDiv);
                
                if (plotElement && plotElement.data) {{
                    const currentData = plotElement.data;
                    
                    for (let traceIndex = 0; traceIndex < currentData.length; traceIndex++) {{
                        const trace = currentData[traceIndex];
                        if (trace.name && trace.name.includes('_highlight')) {{
                            // Hide highlight trace
                            Plotly.restyle(plotDiv, {{visible: [false]}}, [traceIndex]);
                        }}
                    }}
                }}
            }});
        }}
        
        // Event listeners
        document.getElementById('sentenceSelect').addEventListener('change', function() {{
            highlightSentencePath(this.value);
        }});
        
        // Initialize plots when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            initializePlots();
        }});
    </script>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"Interactive scatter plots saved to: {output_path}")


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
    all_dataframes = []
    
    # Generate both 2D and 3D plots
    for is_3d in [False, True]:
        dimension_label = "3D" if is_3d else "2D"
        print(f"Generating {dimension_label} scatter plots with GAE embeddings...")
        gae_plots, gae_results, gae_dataframes = get_scatter_plots_per_model(
            bert_model,
            model_file,
            param_dict,
            label_to_plot="sentence_path",
            reduction_method="tsne",
            is_3d=is_3d
        )
        all_plots.extend(gae_plots)
        all_results.extend(gae_results)
        all_dataframes.extend(gae_dataframes)
        print(f"GAE {dimension_label} results: {gae_results}")
        
        print(f"Generating {dimension_label} scatter plots with TF-IDF embeddings...")
        tfidf_plots, tfidf_results, tfidf_dataframes = get_scatter_plots_per_vectorizer(
            "tfidf",
            label_to_plot="sentence_path",
            reduction_method="tsne",
            is_3d=is_3d
        )
        all_plots.extend(tfidf_plots)
        all_results.extend(tfidf_results)
        all_dataframes.extend(tfidf_dataframes)
        print(f"TF-IDF {dimension_label} results: {tfidf_results}")

        print(f"Generating {dimension_label} scatter plots with trigram embeddings...")
        trigram_plots, trigram_results, trigram_dataframes = get_scatter_plots_per_vectorizer(
            "trigram",
            label_to_plot="sentence_path",
            reduction_method="tsne",
            is_3d=is_3d
        )
        all_plots.extend(trigram_plots)
        all_results.extend(trigram_results)
        all_dataframes.extend(trigram_dataframes)
        print(f"Trigram {dimension_label} results: {trigram_results}")

    # Save combined HTML with interactive features
    output_html_path = os.path.join(output_dir, "clustering_within_scroll_scatter_plots.html")
    save_combined_html(all_plots, all_dataframes, output_html_path)
    
    print("Scatter plot generation completed!")