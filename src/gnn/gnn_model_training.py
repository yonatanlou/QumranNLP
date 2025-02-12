import pickle

import pandas as pd
import torch

from config import BASE_DIR, get_paths_by_domain
from src.baselines.embeddings import (
    VectorizerProcessor,
    get_vectorizer_types,
    get_bert_models,
)

from src.baselines.utils import get_adj_matrix_by_chunks_structure
from src.constants import UNSUPERVISED_METRICS
from src.gnn.hyperparameter_gnn_utils import run_single_gnn_model, run_single_gae_model

DATASET_NAME = "dataset_composition"
IS_SUPERVISED = False
DOMAIN = "dss"

paths = get_paths_by_domain(DOMAIN)
with open(f"{paths['data_path']}/datasets.pkl", "rb") as f:
    processed_datasets = pickle.load(f)

dataset = processed_datasets[DATASET_NAME]

df_origin = pd.read_csv(paths["data_csv_path"])
vectorizer_types = get_vectorizer_types(DOMAIN)

processor = VectorizerProcessor(
    df_origin, paths["processed_vectorizers_path"], vectorizer_types
)
processed_vectorizers = processor.load_or_generate_embeddings()


def train_single_model(is_supervised, param_dict, bert_model, exp_name):
    if is_supervised:
        model, stats_df = run_single_gnn_model(
            processed_vectorizers, dataset, param_dict, verbose=True
        )
    else:
        adjacency_matrix_all = get_adj_matrix_by_chunks_structure(dataset, df_origin)
        model, stats_df = run_single_gae_model(
            adjacency_matrix_all,
            processed_vectorizers,
            dataset,
            param_dict,
            verbose=True,
        )
    model_name_to_save = f"{exp_name}_{param_dict.get('bert_model').split('/')[-1]}"
    torch.save(
        [model.kwargs, model.state_dict()],
        f"{BASE_DIR}/models/{model_name_to_save}.pth",
    )
    print(f"saved model in {BASE_DIR}/models/{model_name_to_save}.pth")
    print(stats_df[UNSUPERVISED_METRICS].round(4).to_dict(orient="records"))
    return stats_df


EXP_NAME = "trained_gae_model"
bert_models = get_bert_models(DOMAIN)
param_dict = {
    "num_adjs": 2,
    "epochs": 50,
    "hidden_dim": 300,
    "latent_dim": 100,  # for GVAE
    "distance": "cosine",
    "learning_rate": 0.001,
    "threshold": 0.99,
    "adjacencies": [
        {"type": "tfidf", "params": {"max_features": 10_000}},
        # {"type": "bert_model", "params": {"model_name":"yonatanlou/BEREL-finetuned-DSS-pos_cls"}},
        {
            "type": "trigram",
            "params": {
                "analyzer": "char",
                "ngram_range": (3, 3),
                "max_features": 10_000,
            },
        },
        # {"type": "fivergram", "params": {"analyzer": "char", "ngram_range": (2, 2),"max_features": 10_000}},
        # {"type": "starr", "params": {}},
    ],
    # "adjacencies": [{"type": "bert_model", "params": {"model_name":"yonatanlou/BEREL-finetuned-DSS-pos_cls"}}],
    "bert_model": "dicta-il/BEREL",
}
meta_params = {"processed_vectorizers": processed_vectorizers, "dataset": dataset}
all_res = []

for bert_model in bert_models:
    param_dict["bert_model"] = bert_model
    tmp_res = train_single_model(IS_SUPERVISED, param_dict, bert_model, EXP_NAME)
    all_res.append(tmp_res)

all_res_df = pd.concat(all_res, axis=0).reset_index(drop=True)
