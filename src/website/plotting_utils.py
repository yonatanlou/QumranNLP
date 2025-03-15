import pandas as pd
import umap
from plotly import express as px
from sklearn.manifold import TSNE


def new_line_each_n_words(txt):
    new_text = ""
    splitted = txt.split(" ")
    word_counter = 0
    for w in splitted:
        new_text += f"{w} "
        if word_counter > 20:
            new_text += "<br>"
            word_counter = 0
        word_counter += 1
    return new_text


def pretty_labels(text):
    if not isinstance(text, str):
        return text

    text = text.replace('_', ' ').replace('.', ' ')
    if len(text)< 6:
        text = text.upper()
    else:
        text = " ".join([t.capitalize() for t in text.split(" ")])

    return text


def plot_embeddings_projection(
    embeddings,
    df_origin,
    method="umap",
    color_by="Section",
    random_state=42,
    save_path=None,
):
    # Parameters for dimensionality reduction
    n_components = 2
    df = df_origin.copy()
    df["text"] = df["text"].apply(new_line_each_n_words)
    df["section"] = df["section"].apply(pretty_labels)
    df["composition"] = df["composition"].apply(pretty_labels)
    # Choose and apply dimensionality reduction method
    if method.lower() == "umap":
        reducer = umap.UMAP(
            random_state=random_state,
            n_neighbors=15,
            min_dist=0.1,
            n_components=n_components,
            metric="cosine",
        )
        reduced_embeddings = reducer.fit_transform(embeddings)
        method_name = "UMAP"

    elif method.lower() == "tsne":
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            init="pca",  # Initialize with PCA for better results
            learning_rate="auto",
            perplexity=30,
            metric="cosine",
        )
        reduced_embeddings = reducer.fit_transform(embeddings)
        method_name = "t-SNE"

    else:
        raise ValueError("Method must be either 'umap' or 'tsne'")

    # Create a DataFrame with the reduced coordinates and book labels
    plot_df = pd.DataFrame(
        {
            "Dim1": reduced_embeddings[:, 0],
            "Dim2": reduced_embeddings[:, 1],
            "Composition": df["composition"],
            "Section": df["section"],
            "Text": df["text"],  # Including text for hover information
            "Chunk path": df["sentence_path"],
        }
    )
    # Create the interactive plot
    fig = px.scatter(
        plot_df,
        x="Dim1",
        y="Dim2",
        color=color_by,
        color_discrete_sequence=px.colors.qualitative.Set3,
        hover_data={
            "Dim1": False,  # Hide Dim1 in hover tooltip
            "Dim2": False,  # Hide Dim2 in hover tooltip
            "Text": False,  # Show text in hover tooltip
            "Composition": True,  # Show book in hover tooltip
            "Section": True,
            "Chunk path": True,
        },
        title=f"{method_name} Visualization of {color_by.replace('Section', 'Sectarian')} Embeddings",
    )

    # Update layout for better visualization
    fig.update_layout(
        width=1000,
        height=800,
        title={"x": 0.5, "xanchor": "center", "font": {"size": 20}},
        legend={
            "title": "Books",
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 1.02,
            "bgcolor": "rgba(255, 255, 255, 0.8)",
        },
        xaxis_title=f"{method_name} Dimension 1",
        yaxis_title=f"{method_name} Dimension 2",
        hovermode="closest",
    )

    # Update traces
    fig.update_traces(marker=dict(size=6), opacity=0.7)

    # Save to HTML if path is provided
    if save_path:
        fig.write_html(save_path)

    return fig
