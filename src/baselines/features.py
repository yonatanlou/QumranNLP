import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from src.constants import BERT_MODELS


def get_starr_features():
    from src.features.Starr.features_keys import feature_list

    return [i[0] for i in feature_list]


def bert_embed(texts, model, tokenizer, method="mean_last_hidden"):
    """
    Extract BERT embeddings using different methods.

    Args:
        texts (list): List of texts to embed.
        model (transformers.PreTrainedModel): Pre-trained BERT model.
        tokenizer (transformers.PreTrainedTokenizer): Corresponding tokenizer.
        method (str): Method to extract embeddings. Options are:
                      "mean_last_hidden" (default) - Mean of the last hidden state.
                      "cls_token" - CLS token embedding.
                      "pooler_output" - Pooler output (if available).
                      "mean_all_layers" - Mean of all hidden layers.

    Returns:
        np.ndarray: Extracted embeddings.
    """
    all_embeddings = []

    for text in tqdm(texts, desc="bert"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)

        if method == "mean_last_hidden":
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        elif method == "cls_token":
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        elif method == "pooler_output":
            embedding = outputs.pooler_output.cpu().numpy()

        elif method == "mean_all_layers":
            all_layers = outputs.hidden_states
            mean_all_layers = torch.stack(all_layers).mean(dim=0)
            embedding = mean_all_layers.mean(dim=1).cpu().numpy()

        else:
            raise ValueError(f"Unknown method: {method}")

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
        assert all(col in df.columns for col in starr_cols)
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
