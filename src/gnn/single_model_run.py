import pickle

import pandas as pd
import torch

from config import BASE_DIR, DATA_PATH
from src.baselines.embeddings import VectorizerProcessor, get_vectorizer_types

import os.path

from src.gnn.hyperparameter_gnn_utils import run_single_gnn_model, run_single_gvae_model

BASELINES_DIR = f"{BASE_DIR}/experiments/baselines"
PROCESSED_VECTORIZERS_PATH = f"{BASELINES_DIR}/processed_vectorizers.pkl"
EXP_NAME = "gvae_model"
DATASET_NAME = "dataset_scroll"
IS_SUPERVISED = False
MODEL_NAME_TO_SAVE = "gvae_model"


with open(f"{BASELINES_DIR}/datasets.pkl", "rb") as f:
    datasets = pickle.load(f)

dataset = datasets[DATASET_NAME]
exp_dir_path = f"{BASE_DIR}/experiments/gnn/{EXP_NAME}"
if not os.path.exists(exp_dir_path):
    os.makedirs(exp_dir_path)
df_origin = pd.read_csv(DATA_PATH)

vectorizer_types = get_vectorizer_types()

processor = VectorizerProcessor(df_origin, PROCESSED_VECTORIZERS_PATH, vectorizer_types)
processed_vectorizers = processor.load_or_generate_embeddings()
param_dict = {
    "num_adjs": 1,
    "epochs": 200,
    "hidden_dim": 300,
    "latent_dim": 768,  # for GVAE
    "distance": "cosine",
    "learning_rate": 0.001,
    "threshold": 0.975,
    "adjacencies": [{"type": "tfidf", "params": {"max_features": 7500}}],
    "bert_model": "yonatanlou/BEREL-finetuned-DSS-maskedLM",
    # "bert_model": "dicta-il/BEREL",
    # "bert_model": "dicta-il/dictabert",
}

if IS_SUPERVISED:
    model, stats_df = run_single_gnn_model(
        processed_vectorizers, dataset, param_dict, verbose=True
    )
else:
    model, stats_df = run_single_gvae_model(
        df_origin, processed_vectorizers, dataset, param_dict, verbose=True
    )
torch.save(
    [model.kwargs, model.state_dict()],
    f"{BASE_DIR}/models/{MODEL_NAME_TO_SAVE}.pth",
)
print(f"saved model in {BASE_DIR}/models/{MODEL_NAME_TO_SAVE}.pth")
