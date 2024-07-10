import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModel

from notebooks.constants import BERT_MODELS


def get_starr_features():
    from src.features.Starr.features_keys import feature_list

    return [i[0] for i in feature_list]


def bert_embed(texts, model, tokenizer):
    all_embeddings = []
    for text in tqdm(texts, desc="bert"):
        # Tokenize the text

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,  # Pad sequences to the length of the longest sequence in the batch
            truncation=True,  # Truncate sequences longer than max_length
            max_length=tokenizer.model_max_length,  # 512
        )
        with torch.no_grad():
            outputs = model(**inputs)

        # Take the mean of the last hidden state for the embedding
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        # print("Shape of last_hidden_state:", outputs.last_hidden_state.shape)
        # print("Shape of pooler_output:", outputs.pooler_output.shape)
        # print("Shape of embedding:", embedding.shape)

        all_embeddings.append(embedding)
    return np.vstack(all_embeddings)


def init_bert_vectorizer(df, text_column, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    vectorizer_matrix = bert_embed(df[text_column].tolist(), model, tokenizer)
    return vectorizer_matrix


def vectorize_text(df, text_column, vectorizer_type):
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer()
        vectorizer_matrix = vectorizer.fit_transform(df[text_column])
    elif vectorizer_type == "trigram":
        vectorizer = CountVectorizer(ngram_range=(3, 3), analyzer="char")
        vectorizer_matrix = vectorizer.fit_transform(df[text_column])
    elif vectorizer_type == "BOW":
        vectorizer = CountVectorizer(analyzer="word")
        vectorizer_matrix = vectorizer.fit_transform(df[text_column])
    elif vectorizer_type == "starr":
        starr_cols = get_starr_features()
        vectorizer_matrix = df[starr_cols].to_numpy()

    elif vectorizer_type in BERT_MODELS:
        model_name = vectorizer_type
        vectorizer_matrix = init_bert_vectorizer(df, text_column, model_name)
    else:
        raise ValueError("Unsupported vectorizer type.")
    return vectorizer_matrix


def get_linkage_matrix(model):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return linkage_matrix


def shorten_path(sentence_path):
    splitted = [i.split(":") for i in sentence_path.split("-")]
    return f"{splitted[0][0]}:{splitted[0][1]}-{splitted[1][0]}"


def create_adjacency_matrix(
    sampled_df, context_similiarity_window, composition_level=True
):
    # Compress the DataFrame only to the required columns
    compressed_df = sampled_df[["original_index", "book", "composition"]]

    # Convert DataFrame columns to numpy arrays for faster access
    original_indices = compressed_df["original_index"].to_numpy()
    books = compressed_df["book"].to_numpy()
    compositions = compressed_df["composition"].to_numpy()

    # Initialize the adjacency matrix
    n = len(compressed_df)
    adjacency_matrix = np.zeros((n, n))
    # Loop to fill the adjacency matrix
    for i in tqdm(range(n), desc="Building adjacency matrix"):
        for j in range(i + 1, n):  # Only compute half since the matrix is symmetric
            if original_indices[i] == original_indices[j]:
                continue
            distance = np.abs(original_indices[i] - original_indices[j])

            if 0 < distance <= context_similiarity_window and books[i] == books[j]:
                adjacency_matrix[i, j] += 1 / distance
                adjacency_matrix[j, i] += 1 / distance

            if (
                composition_level
                and compositions[i] == compositions[j]
                and (compositions[i] is not None)
                and (books[i] != books[j])
            ):
                adjacency_matrix[i, j] += 1
                adjacency_matrix[j, i] += 1

    return adjacency_matrix
