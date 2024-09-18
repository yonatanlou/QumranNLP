# For running hyperparameter tuning for GNN (supervised and unsupervised)
# The BASELINES_DIR will consist the datasets and the embeddings that were created in src.baselines.main

import pickle

import pandas as pd

from config import BASE_DIR, DATA_PATH
from src.baselines.embeddings import VectorizerProcessor, get_vectorizer_types
from src.gnn.hyperparameter_gnn_utils import run_gnn_exp
from src.gnn.utils import generate_parameter_combinations
import os.path


BASELINES_DIR = f"{BASE_DIR}/experiments/baselines"
PROCESSED_VECTORIZERS_PATH = f"{BASELINES_DIR}/processed_vectorizers.pkl"
EXP_NAME = "gvae_init"
NUM_COMBINED_GRAPHS = 2
OVERWRITE = True
IS_SUPERVISED = False  # regular GCN for supervised, GVAE for unsupervised

with open(f"{BASELINES_DIR}/datasets.pkl", "rb") as f:
    datasets = pickle.load(f)

PARAMS = {
    "epochs": [500],
    "hidden_dims": [300, 500],
    "latent_dims": [100],  # only for GVAE
    "distances": ["cosine"],
    "learning_rates": [0.001],
    "thresholds": [0.99],
    "bert_models": [
        "dicta-il/BEREL",
        "dicta-il/dictabert",
        "onlplab/alephbert-base",
        "yonatanlou/BEREL-finetuned-DSS-maskedLM",
        "yonatanlou/alephbert-base-finetuned-DSS-maskedLM",
        "yonatanlou/dictabert-finetuned-DSS-maskedLM",
    ],
    "adj_types": {
        "tfidf": {"max_features": 7500},
        "trigram": {"analyzer": "char", "ngram_range": (3, 3)},
        "BOW-n_gram": {"analyzer": "word", "ngram_range": (1, 1)},
        "starr": {},
        # "bert-berel": {"type": "dicta-il/BEREL"},
        # "bert-alephbert": {"type": "onlplab/alephbert-base"},
        # "bert-finetune-lm": {"type": "yonatanlou/BEREL-finetuned-DSS-maskedLM"},
    },
}

all_param_dicts = generate_parameter_combinations(PARAMS, NUM_COMBINED_GRAPHS)


for dataset_name, dataset in datasets.items():
    print(f"starting with {dataset_name}")
    exp_dir_path = f"{BASE_DIR}/experiments/gnn/{EXP_NAME}"
    if not os.path.exists(exp_dir_path):
        os.makedirs(exp_dir_path)
    file_name = (
        f"{exp_dir_path}/{EXP_NAME}_{dataset.label}_{NUM_COMBINED_GRAPHS}_adj_types.csv"
    )
    if os.path.isfile(file_name) and not OVERWRITE:
        continue
    df = pd.read_csv(DATA_PATH)
    vectorizer_types = get_vectorizer_types()

    processor = VectorizerProcessor(df, PROCESSED_VECTORIZERS_PATH, vectorizer_types)
    processed_vectorizers = processor.load_or_generate_embeddings()
    df = df.reset_index()

    run_gnn_exp(
        all_param_dicts,
        df,
        processed_vectorizers,
        file_name,
        dataset,
        IS_SUPERVISED,
        verbose=True,
    )
