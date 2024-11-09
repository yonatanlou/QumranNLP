from datetime import datetime

import pandas as pd
from tqdm import tqdm
import torch

from src.baselines.utils import create_adjacency_matrix
from src.gnn.adjacency import AdjacencyMatrixGenerator, CombinedAdjacencyMatrixGenerator
from src.gnn.model import GCN, train, train_gvae, GVAE, unsupervised_evaluation
from src.gnn.utils import get_data_object


def run_single_gnn_model(processed_vectorizers, dataset, param_dict, verbose=False):
    df = dataset.df
    masks = {
        "train_mask": dataset.train_mask,
        "val_mask": dataset.val_mask,
        "test_mask": dataset.test_mask,
    }
    meta_params = {"processed_vectorizers": processed_vectorizers, "dataset": dataset}

    print(f"{datetime.now()} - started - {param_dict}")
    edge_index, edge_attr, adj_matrix = create_adj_matrix_for_pytorch_geometric(
        df, param_dict, meta_params
    )
    X = processed_vectorizers[param_dict["bert_model"]]
    X = X[dataset.relevant_idx_to_embeddings]
    data, label_encoder = get_data_object(
        X, df, dataset.label, edge_index, edge_attr, masks
    )
    gcn = GCN(
        data.num_features,
        param_dict["hidden_dim"],
        data.num_classes,
        param_dict["learning_rate"],
    )

    # Train the GCN
    gcn, stats = train(
        gcn,
        data,
        param_dict["epochs"],
        patience=param_dict["epochs"] / 3,
        verbose=verbose,
    )
    stats_df = pd.DataFrame(stats)
    for param, value in param_dict.items():
        if param == "adjacencies":
            continue
        stats_df[param] = value
    adj_types_str = " & ".join([adj["type"] for adj in param_dict["adjacencies"]])
    stats_df["adj_type"] = adj_types_str
    stats_df["num_edges"] = edge_attr.shape[0]
    return gcn, stats_df


def create_adj_matrix_for_pytorch_geometric(df, param_dict, meta_params):
    if param_dict["num_adjs"] == 1:
        adj_info = param_dict["adjacencies"][0]
        adj_gen = AdjacencyMatrixGenerator(
            vectorizer_type=adj_info["type"],
            vectorizer_params=adj_info["params"],
            threshold=param_dict["threshold"],
            distance_metric=param_dict["distance"],
            meta_params=meta_params,
            normalize=True,
        )

        edge_index, edge_attr, adj_matrix = adj_gen.generate_graph(df)

    else:  # combining more than one graphs together
        adj_generators = []
        for adj_info in param_dict["adjacencies"]:
            adj_gen = AdjacencyMatrixGenerator(
                vectorizer_type=adj_info["type"],
                vectorizer_params=adj_info["params"],
                threshold=param_dict["threshold"],
                distance_metric=param_dict["distance"],
                meta_params=meta_params,
                normalize=False,
            )
            adj_generators.append(adj_gen)

        combined_generator = CombinedAdjacencyMatrixGenerator(
            adj_generators,
            combine_method="add",
            threshold=param_dict["threshold"],
            normalize=True,
        )
        edge_index, edge_attr, adj_matrix = combined_generator.combine_graphs(
            [df] * param_dict["num_adjs"]
        )
    return edge_index, edge_attr, adj_matrix


def run_single_gvae_model(
    origin_df, processed_vectorizers, dataset, param_dict, verbose=False
):
    df = dataset.df
    masks = {
        "train_mask": dataset.train_mask,
        "val_mask": dataset.val_mask,
        "test_mask": dataset.test_mask,
    }
    meta_params = {"processed_vectorizers": processed_vectorizers, "dataset": dataset}
    edge_index, edge_attr, adj_matrix = create_adj_matrix_for_pytorch_geometric(
        df, param_dict, meta_params
    )
    print(f"{datetime.now()} - started - {param_dict}")

    X = processed_vectorizers[param_dict["bert_model"]]
    X = X[dataset.relevant_idx_to_embeddings]
    data, label_encoder = get_data_object(
        X, df, dataset.label, edge_index, edge_attr, masks
    )
    gvae = GVAE(
        data.num_features,
        param_dict["hidden_dim"],
        param_dict["latent_dim"],
    )
    optimizer = torch.optim.Adam(
        gvae.parameters(), lr=param_dict["learning_rate"], weight_decay=5e-4
    )

    adjacency_matrix_all = create_adjacency_matrix(
        origin_df,
        context_similiarity_window=3,
        composition_level=False,
    )
    adjacency_matrix_tmp = adjacency_matrix_all[dataset.relevant_idx_to_embeddings, :][
        :, dataset.relevant_idx_to_embeddings
    ]

    gvae, stats = train_gvae(
        gvae,
        data,
        optimizer,
        param_dict["epochs"],
        dataset,
        adjacency_matrix_tmp,

        verbose=verbose,
    )


    stats_df = pd.DataFrame(stats)
    for param, value in param_dict.items():
        if param == "adjacencies":
            continue
        stats_df[param] = value
    adj_types_str = " & ".join([adj["type"] for adj in param_dict["adjacencies"]])
    stats_df["adj_type"] = adj_types_str
    stats_df["num_edges"] = edge_attr.shape[0]
    return gvae, stats_df


def run_gnn_exp(
    all_param_dicts,
    df,
    processed_vectorizers,
    file_name,
    dataset,
    is_supervised,
    verbose=False,
):
    print(f"{datetime.now()} - started")
    final_results = []
    metric = ""
    for param_dict in tqdm(all_param_dicts, desc="Parameter Combinations"):
        if is_supervised:
            model, stats_df = run_single_gnn_model(
                processed_vectorizers, dataset, param_dict, verbose=verbose
            )
            metric = "test_acc"

        else:
            model, stats_df = run_single_gvae_model(
                df, processed_vectorizers, dataset, param_dict, verbose=verbose
            )
            metric = "dasgupta"
        final_results.append(stats_df)
    final_df = pd.concat(final_results).sort_values(by=[metric], ascending=False)

    final_df.to_csv(
        file_name,
        index=False,
    )
    print(f"{datetime.now()} - finished, saved to {file_name}")
