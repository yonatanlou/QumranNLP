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


#
#
# adj_gen = AdjacencyMatrixGenerator(
#     vectorizer_type="tfidf",
#     vectorizer_params={"max_features": 7500},
#     threshold=0.99,
#     distance_metric="cosine",
#     meta_params=None,
#     normalize=True,
# )
# df = df[df["book"].isin(["1QM","1QSa"])]
# edge_index, edge_attr, adj_matrix = adj_gen.generate_graph(df)
# X = processed_vectorizers[param_dict["bert_model"]]
# X = X[df.index]
# X = X.astype("float32")
# X_tensor = torch.FloatTensor(X)
# with torch.no_grad():
#     h, logits = gcn(X_tensor, edge_index, edge_attr)
# node_embeddings_np = h.numpy()
#
# # Create a DataFrame with node embeddings
# embeddings_df = pd.DataFrame(
#     node_embeddings_np,
#     index=df.index,
#     columns=[f"embedding_{i}" for i in range(node_embeddings_np.shape[1])]
# )
#
# # Merge embeddings with original DataFrame
# result_df = pd.concat([df, embeddings_df], axis=1)
