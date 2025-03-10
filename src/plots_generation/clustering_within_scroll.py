import os

import pandas as pd
import torch

from config import BASE_DIR, get_paths_by_domain
from src.baselines.embeddings import VectorizerProcessor, get_bert_models
from src.baselines.utils import create_adjacency_matrix, set_seed_globally
from src.gnn.adjacency import AdjacencyMatrixGenerator
from src.plots_generation.analysis_utils import (
    hirerchial_clustering_by_scroll_gnn,
    get_gae_embeddings,
)
from src.plots_generation.scrolls_labeling import (
    labels_1QS,
    labels_hodayot,
    labels_1QM,
)

exp_dir = f"{BASE_DIR}/experiments/dss/clustering_by_scroll"
DOMAIN = "dss"
paths = get_paths_by_domain(DOMAIN)
DATA_CSV_PATH = paths["data_csv_path"]

df = pd.read_csv(DATA_CSV_PATH)


vectorizers = get_bert_models(DOMAIN) + ["trigram", "tfidf", "starr"]
processor = VectorizerProcessor(df, paths["processed_vectorizers_path"], vectorizers)
PROCESSED_VECTORIZERS = processor.load_or_generate_embeddings()
ADJACENCY_MATRIX_ALL = create_adjacency_matrix(
    df,
    context_similiarity_window=3,
    composition_level=False,
)


def get_metrics_per_model(
    bert_model: str, model_file: str, param_dict: dict, label_to_plot, path_to_save=None
) -> list:
    # Predefined paths and configurations
    set_seed_globally()
    models_dir = f"{BASE_DIR}/models"
    data_csv_path = paths["data_csv_path"]
    path_to_save = path_to_save + "/GNN_{}.pdf" if path_to_save else None
    # Default parameter dictionary
    param_dict["bert_model"] = bert_model

    # Read and filter DataFrame
    df = pd.read_csv(data_csv_path)
    df_ = df[df["book"].isin(["1QM", "1QSa", "1QS", "1QHa"])]
    embeddings_gvae = get_gae_embeddings(
        df_, PROCESSED_VECTORIZERS, bert_model, model_file, param_dict
    )
    # Labels setup
    labels_all = {
        # "['1QM']": labels_1QM,
        "['1QHa']": labels_hodayot,
        # "['1QS', '1QSa']": labels_1QS,
    }

    # Calculate metrics for each scroll
    all_results = []
    for scroll, labels in labels_all.items():
        metrics_gnn = hirerchial_clustering_by_scroll_gnn(
            df_,
            eval(scroll),
            labels,
            f"Unsupervised Clustering of the Hodayot Composition",
            embeddings_gvae,
            ADJACENCY_MATRIX_ALL,
            label_to_plot,
            {"linkage_m": "ward", "color_threshold": 0.7},
            path_to_save.format(eval(scroll)) if path_to_save else None,
        )

        # Add metadata to results
        metrics_gnn.update(
            {
                "scroll": scroll,
                "vectorizer": bert_model,
                "model_file": model_file,
            }
        )

        all_results.append(metrics_gnn)

    return all_results


if __name__ == "__main__":
    # best by Hodayot
    param_dict = {
        "num_adjs": 2,
        "epochs": 50,
        "hidden_dim": 300,
        "distance": "cosine",
        "learning_rate": 0.001,
        "threshold": 0.9,
        "adjacencies": [
            {"type": "tfidf", "params": {"max_features": 1000, "min_df": 0.001}},
            {
                "type": "trigram",
                "params": {
                    "analyzer": "char",
                    "ngram_range": (3, 3),
                    "min_df": 0.001,
                    "max_features": 1000,
                },
            },
        ],
    }
    bert_model = "dicta-il/BEREL"
    model_file = "trained_gae_model_BEREL.pth"
    path_to_save = f"{BASE_DIR}/reports/plots"
    result = get_metrics_per_model(
        bert_model,
        model_file,
        param_dict,
        label_to_plot="sentence_path",
        path_to_save=path_to_save,
    )

    # all_res_sim = []
    # for threshold in [0.8, 0.9,0.95, 0.99]:
    #     param_dict["threshold"] = threshold
    #     for bert_model in get_bert_models(DOMAIN):
    #         for model_file in ['trained_gae_model_BEREL-finetuned-DSS-maskedLM.pth',
    #                            'trained_gae_model_BEREL.pth',
    #                            ]:
    #             for max_f in [1000,5000,10000]:
    #                 param_dict["adjacencies"][0]["params"]["max_features"] = max_f
    #                 param_dict["adjacencies"][1]["params"]["max_features"] = max_f
    #                 for min_df in [0.001,0.005,0.01]:
    #                     param_dict["adjacencies"][0]["params"]["min_df"] = min_df
    #                     param_dict["adjacencies"][1]["params"]["min_df"] = min_df
    #
    #                     result = get_metrics_per_model(
    #                         bert_model,
    #                         model_file,
    #                         param_dict,
    #                         label_to_plot=None,
    #                         path_to_save=None
    #                     )
    #                     result[0].update({"threshold": threshold, "bert_model": bert_model, "model_file": model_file, "min_df":min_df,"max_f": max_f})
    #                     all_res_sim.append(result[0])
    #                     print(result[0])
    #                     print()
    # pd.DataFrame(all_res_sim).to_csv(
    #     f"/Users/yonatanlou/dev/QumranNLP/experiments/dss/clustering_by_scroll/clustering_within_composition_comp.csv",
    #     index=False)
