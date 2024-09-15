import pickle

import pandas as pd
import torch

from config import BASE_DIR
from src.baselines.embeddings import VectorizerProcessor, get_vectorizer_types
from src.gnn.adjacency import AdjacencyMatrixGenerator

import os.path

from src.gnn.hyperparameter_gnn_utils import run_single_gnn_model

data_path = "/Users/yonatanlou/dev/QumranNLP/data/processed_data/filtered_df_CHUNK_SIZE=100_MAX_OVERLAP=15_PRE_PROCESSING_TASKS=[]_2024_02_09.csv"
results_dir = f"{BASE_DIR}/experiments/baselines"
PROCESSED_VECTORIZERS_PATH = (
    f"{BASE_DIR}/experiments/baselines/processed_vectorizers.pkl"
)
EXP_NAME = "gcn_model"

with open(f"{results_dir}/datasets.pkl", "rb") as f:
    datasets = pickle.load(f)

dataset_name = "dataset_scroll"
dataset = datasets[dataset_name]
exp_dir_path = f"{BASE_DIR}/experiments/gnn/{EXP_NAME}"
if not os.path.exists(exp_dir_path):
    os.makedirs(exp_dir_path)

df = dataset.df
vectorizer_types = get_vectorizer_types()

processor = VectorizerProcessor(df, PROCESSED_VECTORIZERS_PATH, vectorizer_types)
processed_vectorizers = processor.load_or_generate_embeddings()
df = df.reset_index()
param_dict = {
    "num_adjs": 1,
    "epochs": 500,
    "hidden_dim": 300,
    "distance": "cosine",
    "learning_rate": 0.001,
    "threshold": 0.99,
    "adjacencies": [{"type": "tfidf", "params": {"max_features": 7500}}],
    "bert_model": "yonatanlou/BEREL-finetuned-DSS-maskedLM",
}


gcn, stats_df = run_single_gnn_model(
    df, processed_vectorizers, dataset, param_dict, verbose=True
)
torch.save(
    [gcn.kwargs, gcn.state_dict()],
    f"{BASE_DIR}/models/gcn_model_train_on_scroll.pth",
)
print("saved model in {BASE_DIR}/models/gcn_model_train_on_scroll.pth")

# from src.gnn.model import GCN
# import torch
#
# param_dict = {
#     "num_adjs": 1,
#     "epochs": 500,
#     "hidden_dim": 300,
#     "distance": "cosine",
#     "learning_rate": 0.001,
#     "threshold": 0.98,
#     "adjacencies": [{"type": "tfidf", "params": {"max_features": 7500}}],
#     "bert_model": "yonatanlou/BEREL-finetuned-DSS-maskedLM",
# }
#
# GNN_MODEL_PATH = f"{BASE_DIR}/models/gcn_model_train_on_scroll.pth"
# kwargs, state = torch.load(GNN_MODEL_PATH)
# model = GCN(**kwargs)
# model.load_state_dict(state)
# print("")
