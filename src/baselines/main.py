import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from config import BASE_DIR
from src.baselines.utils import create_adjacency_matrix, set_seed_globally
from src.baselines.create_datasets import QumranDataset, save_dataset_for_finetuning
from src.baselines.embeddings import get_vectorizer_types, VectorizerProcessor
from src.baselines.ml import evaluate_unsupervised_metrics, evaluate_supervised_metrics


MODELS = [
    LogisticRegression(max_iter=500),
    LinearSVC(),
    KNeighborsClassifier(),
    MLPClassifier(random_state=42, max_iter=500),
]


def make_baselines_results(
    data_path,
    processed_vectorizers_path,
    results_dir,
    train_frac,
    val_frac,
    tasks=["scroll", "composition", "sectarian"],
):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    df = pd.read_csv(data_path)

    vectorizers = get_vectorizer_types()
    processor = VectorizerProcessor(df, processed_vectorizers_path, vectorizers)
    processed_vectorizers = processor.load_or_generate_embeddings()
    set_seed_globally()
    adjacency_matrix_all = create_adjacency_matrix(
        df,
        context_similiarity_window=3,
        composition_level=False,
    )

    dataset_composition = QumranDataset(
        df, "composition", train_frac, val_frac, processed_vectorizers
    )
    dataset_scroll = QumranDataset(
        df, "book", train_frac, val_frac, processed_vectorizers
    )
    dataset_sectarian = QumranDataset(
        df, "section", train_frac, val_frac, processed_vectorizers
    )
    datasets = {
        "dataset_composition": dataset_composition,
        "dataset_scroll": dataset_scroll,
        "dataset_sectarian": dataset_sectarian,
    }
    # for dataset_name, dataset in datasets.items():
    #     save_dataset_for_finetuning(f"{results_dir}/{dataset_name}.pkl", dataset)
    with open(f"{results_dir}/datasets.pkl", "wb") as f:
        pickle.dump(datasets, f)

    datasets = {k: v for k, v in datasets.items() if k.split("_")[1] in tasks}
    for dataset_name, dataset in datasets.items():
        set_seed_globally()
        print(f"calculating metrics for {dataset_name}")
        evaluate_unsupervised_metrics(
            adjacency_matrix_all, dataset, vectorizers, results_dir
        )
        evaluate_supervised_metrics(MODELS, vectorizers, dataset, results_dir)


if __name__ == "__main__":
    data_path = f"{BASE_DIR}/data/processed_data/filtered_df_CHUNK_SIZE=100_MAX_OVERLAP=15_PRE_PROCESSING_TASKS=[]_2024_02_09.csv"
    results_dir = f"{BASE_DIR}/experiments/baselines"
    processed_vectorizers_path = (
        f"{results_dir}/processed_vectorizers.pkl"
    )

    make_baselines_results(
        data_path,
        processed_vectorizers_path,
        results_dir,
        train_frac=0.7,
        val_frac=0.1,
        tasks=["scroll", "composition", "sectarian"],
    )
