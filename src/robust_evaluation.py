import pandas as pd
from tqdm import tqdm

from config import get_paths_by_domain, BASE_DIR
from base_utils import measure_time
from src.baselines.create_datasets import create_dss_datasets
from src.baselines.embeddings import VectorizerProcessor, get_vectorizer_types
from src.baselines.ml import evaluate_unsupervised_metrics
from src.baselines.utils import create_adjacency_matrix

from src.gnn.hyperparameter_gnn_utils import run_gnn_exp
from src.gnn.utils import (
    generate_parameter_combinations,
    create_gnn_params_cv,
)
import os.path


@measure_time
def run_gae_for_cv_evaulation(
    dataset, domain, num_combined_graphs, is_supervised, df_frac_remove=0, seed=42
) -> pd.DataFrame:
    paths = get_paths_by_domain(domain)
    processed_datasets, _ = create_dss_datasets(
        ["scroll", "composition", "sectarian"],
        0.7,
        0.1,
        paths,
        get_vectorizer_types(domain),
        df_frac_remove=df_frac_remove,
        seed=seed,
        save_dataset=False,
    )

    dataset_info = processed_datasets[dataset]
    print(f"Starting with {dataset}")

    df = pd.read_csv(paths["data_csv_path"])
    vectorizer_types = get_vectorizer_types(domain)
    processor = VectorizerProcessor(
        df, paths["processed_vectorizers_path"], vectorizer_types
    )
    processed_vectorizers = processor.load_or_generate_embeddings()
    df = df.reset_index()
    params = create_gnn_params_cv(domain, is_supervised)
    all_param_dicts = generate_parameter_combinations(params, num_combined_graphs)
    iter_df = run_gnn_exp(
        all_param_dicts,
        df,
        processed_vectorizers,
        None,
        dataset_info,
        is_supervised,
        verbose=False,
    )
    return iter_df


def run_gae_n_iters(
    n_iters,
    domain="dss",
    dataset="scroll",
    is_supervised=False,
    df_frac_remove=0,
    file_name=None,
    num_combined_graphs=1,
):
    all_results = []
    for iter in tqdm(range(n_iters)):
        iter_df = run_gae_for_cv_evaulation(
            dataset,
            domain,
            num_combined_graphs,
            is_supervised,
            df_frac_remove=df_frac_remove,
            seed=iter,
        )
        iter_df["iter"] = iter
        all_results.append(iter_df)

    final_df = pd.concat(all_results)

    if file_name is not None:
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        final_df.to_csv(file_name, index=False)
        print(f"Saved results to {file_name}")


def run_unsupervised_baselines(dataset, domain, df_frac_remove=0.1, seed=42):
    paths = get_paths_by_domain(domain)
    processed_datasets, _ = create_dss_datasets(
        [dataset.split("_")[1]],
        0.7,
        0.1,
        paths,
        get_vectorizer_types(domain),
        df_frac_remove=df_frac_remove,
        seed=seed,
        save_dataset=False,
    )

    dataset = processed_datasets[dataset]

    df_origin = pd.read_csv(paths["data_csv_path"])
    adjacency_matrix_all = create_adjacency_matrix(
        df_origin,
        context_similiarity_window=3,
        composition_level=False,
    )
    vectorizers = get_vectorizer_types(domain)
    metrics_df = evaluate_unsupervised_metrics(
        adjacency_matrix_all, dataset, vectorizers, None
    )
    return metrics_df


def run_unsupervised_baselines_n_iters(
    n_iters, file_name, domain="dss", dataset="dataset_scroll", df_frac_remove=0.1
):
    all_results = []
    for iter in tqdm(range(n_iters)):
        iter_df = run_unsupervised_baselines(
            dataset, domain, df_frac_remove=df_frac_remove, seed=iter
        )
        iter_df["iter"] = iter
        all_results.append(iter_df)

    final_df = pd.concat(all_results)

    if file_name is not None:
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        final_df.to_csv(file_name, index=False)
        print(f"Saved results to {file_name}")


if __name__ == "__main__":
    df_frac_remove = 0.1
    assert df_frac_remove < 1
    n_iters = int(df_frac_remove**-1)

    exp_dir = f"{BASE_DIR}/experiments/dss/cross_validation"
    for label in ["scroll", "composition"]:
        run_unsupervised_baselines_n_iters(
            n_iters,
            file_name=f"{exp_dir}/baselines/{label.replace('scroll', 'book')}_unsupervised.csv",
            domain="dss",
            dataset=f"dataset_{label}",
            df_frac_remove=df_frac_remove,
        )
        run_gae_n_iters(
            n_iters,
            domain="dss",
            dataset=f"dataset_{label}",
            is_supervised=False,
            df_frac_remove=df_frac_remove,
            file_name=f"{exp_dir}/gnn/gae_init/gae_init_{label.replace('scroll', 'book')}_2_adj_types.csv",
            num_combined_graphs=2,
        )
