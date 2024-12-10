# For running hyperparameter tuning for GNN (supervised and unsupervised)

import click
import pickle
import pandas as pd
from config import get_paths_by_domain
from base_utils import measure_time
from src.baselines.embeddings import VectorizerProcessor, get_vectorizer_types
from src.constants import (
    DSS_OPTIONAL_DATASETS,
)
from src.gnn.hyperparameter_gnn_utils import run_gnn_exp
from src.gnn.utils import (
    generate_parameter_combinations,
    create_gnn_params,
    extract_datasets,
)
import os.path


@click.command()
@click.option(
    "--datasets",
    default="all",
    help=f"Dataset name to run the experiment on (for dss one of={DSS_OPTIONAL_DATASETS}",
)
@click.option(
    "--domain",
    type=click.Choice(["dss", "bible"], case_sensitive=False),
    multiple=False,
    default="dss",
    help="dss or bible",
)
@click.option(
    "--num-combined-graphs",
    type=int,
    default=2,
    help="Number of combined graph types",
)
@click.option("--exp-name", default="gcn_init", help="Experiment name")
@click.option(
    "--results-dir",
    default="experiments/dss/gnn",
    help="Directory to store the results",
)
@click.option(
    "--is_supervised",
    is_flag=True,
    help="Run supervised GCN instead of unsupervised GVAE (empty for unsupervised)",
)
@measure_time
def run_gnn_exp_main(
    datasets, domain, num_combined_graphs, exp_name, results_dir, is_supervised
):
    # print all params:
    print("run params:")
    print(f"dataset: {datasets}")
    print(f"domain: {domain}")
    print(f"num_combined_graphs: {num_combined_graphs}")
    print(f"exp_name: {exp_name}")
    print(f"results_dir: {results_dir}")
    print(f"is_supervised: {is_supervised}")
    datasets = extract_datasets(datasets, domain)

    for dataset in datasets:
        run_gnn_hyperparameter_tuning(
            dataset,
            domain,
            num_combined_graphs,
            exp_name,
            results_dir,
            is_supervised,
        )


@measure_time
def run_gnn_hyperparameter_tuning(
    dataset, domain, num_combined_graphs, exp_name, results_dir, is_supervised
):
    """
    Run hyperparameter tuning for GNN (supervised and unsupervised).

    The RESULTS_DIR will consist the datasets and the embeddings that were created in src.baselines.main.
    """
    paths = get_paths_by_domain(domain)
    with open(f"{paths['data_path']}/datasets.pkl", "rb") as f:
        processed_datasets = pickle.load(f)

    dataset_info = processed_datasets[dataset]
    print(f"Starting with {dataset}")
    exp_dir_path = f"{results_dir}/{exp_name}"
    if not os.path.exists(exp_dir_path):
        os.makedirs(exp_dir_path)
    file_name = f"{exp_dir_path}/{exp_name}_{dataset_info.label}_{num_combined_graphs}_adj_types.csv"
    if os.path.isfile(file_name) and not OVERWRITE:
        click.echo(
            f"Skipping {dataset} as the file already exists and OVERWRITE is False."
        )
        return

    df = pd.read_csv(paths["data_csv_path"])
    vectorizer_types = get_vectorizer_types(domain)
    processor = VectorizerProcessor(
        df, paths["processed_vectorizers_path"], vectorizer_types
    )
    processed_vectorizers = processor.load_or_generate_embeddings()
    df = df.reset_index()
    params = create_gnn_params(domain, is_supervised)
    all_param_dicts = generate_parameter_combinations(params, num_combined_graphs)
    run_gnn_exp(
        all_param_dicts,
        df,
        processed_vectorizers,
        file_name,
        dataset_info,
        is_supervised,
        verbose=False,
    )


OVERWRITE = True

if __name__ == "__main__":
    run_gnn_exp_main()
