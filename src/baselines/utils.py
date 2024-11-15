import random

import numpy as np
import torch
from tqdm import tqdm


def create_adjacency_matrix(
    sampled_df, context_similiarity_window, composition_level=True
):
    # Compress the DataFrame only to the required columns
    sampled_df["original_index"] = range(len(sampled_df))
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


def set_seed_globally(seed=42):
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def calculate_jaccard_unsupervised(labels_true, labels_pred):
    """
    Calculates the Jaccard index between true and predicted labels without sklearn.

    Args:
        labels_true: True cluster labels.
        labels_pred: Predicted cluster labels.

    Returns:
        Jaccard index.
    """

    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    # Create pairwise comparison matrices
    same_in_true = labels_true[:, None] == labels_true[None, :]
    same_in_pred = labels_pred[:, None] == labels_pred[None, :]

    # Calculate intersection and union of the matrices
    intersection = np.logical_and(same_in_true, same_in_pred).sum() - len(labels_true)
    union = np.logical_or(same_in_true, same_in_pred).sum() - len(labels_true)

    return intersection / union if union != 0 else 0.0
