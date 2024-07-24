import os
import pickle

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from config import BASE_DIR
from notebooks.features import create_adjacency_matrix
from notebooks.notebooks_utils import set_seed_globally
from src.baselines.create_datasets import QumranDataset, save_dataset_for_finetuning
from src.baselines.embeddings import get_vectorizer_types, VectorizerProcessor
from src.baselines.ml import evaluate_unsupervised_metrics, evaluate_supervised_metrics


TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
MODELS = [
    RandomForestClassifier(random_state=42, n_estimators=150),
    LogisticRegression(max_iter=500),
    LinearSVC(),
    KNeighborsClassifier(),
    AdaBoostClassifier(random_state=42, n_estimators=100),
    MLPClassifier(random_state=42, max_iter=500),
]
CHUNK_SIZE = 100
DATA_PATH = f"{BASE_DIR}/notebooks/data/filtered_text_and_starr_features_{CHUNK_SIZE}_words_nonbib_17_06_2024.csv"
PROCESSED_VECTORIZERS_PATH = f"{BASE_DIR}/src/data/processed_vectorizers.pkl"
BASELINE_DIR = f"{BASE_DIR}/reports/baselines"
if not os.path.exists(BASELINE_DIR):
    os.makedirs(BASELINE_DIR)

df = pd.read_csv(DATA_PATH)

vectorizers = get_vectorizer_types()
processor = VectorizerProcessor(df, PROCESSED_VECTORIZERS_PATH, vectorizers)
processed_vectorizers = processor.load_or_generate_embeddings()
set_seed_globally()

dataset_composition = QumranDataset(
    df, "composition", TRAIN_FRAC, VAL_FRAC, processed_vectorizers
)
dataset_scroll = QumranDataset(df, "book", TRAIN_FRAC, VAL_FRAC, processed_vectorizers)
dataset_sectarian = QumranDataset(
    df, "section", TRAIN_FRAC, VAL_FRAC, processed_vectorizers
)
datasets = {
    "dataset_composition": dataset_composition,
    "dataset_scroll": dataset_scroll,
    "dataset_sectarian": dataset_sectarian,
}

for dataset_name, dataset in datasets.items():
    save_dataset_for_finetuning(f"{BASE_DIR}/src/data/{dataset_name}.pkl", dataset)
    print(f"Saved dataset for finetuning {dataset}")
with open(f"{BASE_DIR}/src/data/datasets.pkl", "wb") as f:
    pickle.dump(datasets, f)

adjacency_matrix_all = create_adjacency_matrix(
    df,
    context_similiarity_window=3,
    composition_level=True,
)

for dataset_name, dataset in datasets.items():
    set_seed_globally()
    print(f"calculating metrics for {dataset_name}")
    metrics_df_unsupervised = evaluate_unsupervised_metrics(
        adjacency_matrix_all, dataset, vectorizers, BASELINE_DIR
    )
    metrics_df_supervised = evaluate_supervised_metrics(
        MODELS, vectorizers, dataset, BASELINE_DIR
    )
# TODO add data from fine tuned models.
