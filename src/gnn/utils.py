import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import numpy as np


def create_param_dict(n_adjs, adj_combinations, meta_params):
    param_dicts = []
    for adjs in adj_combinations:
        param_dict = {
            "num_adjs": n_adjs,
            "epochs": meta_params["epochs"],
            "hidden_dim": meta_params["hidden_dim"],
            "latent_dim": meta_params["latent_dim"],
            "distance": meta_params["distance"],
            "learning_rate": meta_params["learning_rate"],
            "threshold": meta_params["threshold"],
            "adjacencies": [{"type": adj[0], "params": adj[1]} for adj in adjs],
            "bert_model": meta_params["bert_model"],
        }
        param_dicts.append(param_dict)
    return param_dicts


def get_data_object(X, df, label_column, edge_index, edge_attr, masks):
    X = X.astype("float32")
    y = df[label_column]

    label_encoder = LabelEncoder()
    y_numeric = label_encoder.fit_transform(y)

    train_mask, val_mask, test_mask = (
        masks["train_mask"],
        masks["val_mask"],
        masks["test_mask"],
    )

    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(y_numeric, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_features=X.shape[1],
        num_classes=len(np.unique(y_numeric)),
    )

    print(data)

    return data, label_encoder


from itertools import product, combinations


def generate_parameter_combinations(params, num_combined_graphs):
    all_param_dicts = []

    meta_param_combinations = product(
        params["epochs"],
        params["thresholds"],
        params["distances"],
        params["hidden_dims"],
        params["latent_dims"],
        params["learning_rates"],
        params["bert_models"],
    )

    for (
        epoch,
        threshold,
        distance,
        hidden_dim,
        latent_dim,
        lr,
        bert_model,
    ) in meta_param_combinations:
        meta_params = {
            "epochs": epoch,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "distance": distance,
            "learning_rate": lr,
            "threshold": threshold,
            "bert_model": bert_model,
        }
        for n in range(1, num_combined_graphs + 1):
            adj_combinations = combinations(params["adj_types"].items(), n)
            all_param_dicts.extend(create_param_dict(n, adj_combinations, meta_params))

    return all_param_dicts
