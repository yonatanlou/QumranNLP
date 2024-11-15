# For running hyperparameter tuning for GNN (supervised and unsupervised)

import click
import pickle
import pandas as pd
from config import BASE_DIR, get_paths_by_domain
from base_utils import measure_time
from src.baselines.embeddings import VectorizerProcessor, get_vectorizer_types
from src.constants import DSS_OPTIONAL_DATASETS, BIBLE_OPTIONAL_DATASETS
from src.gnn.hyperparameter_gnn_utils import run_gnn_exp
from src.gnn.utils import generate_parameter_combinations
import os.path


EXP_NAME = "gcn_init"
NUM_COMBINED_GRAPHS = 1
OVERWRITE = True
IS_SUPERVISED = False  # regular GCN for supervised, GVAE for unsupervised
GNN_EXP_RESULTS_DIR = f"{BASE_DIR}/dss/experiments/gnn"
PARAMS = {
    "epochs": [250],
    "hidden_dims": [
        300,
        # 500
    ],
    "latent_dims": [100],  # only for GVAE
    "distances": ["cosine"],
    "learning_rates": [0.001],
    "thresholds": [0.99],
    "bert_models": [
        "dicta-il/BEREL",
        "dicta-il/dictabert",
        # "onlplab/alephbert-base",
        # "dicta-il/MsBERT",
        "yonatanlou/BEREL-finetuned-DSS-maskedLM",
        # "yonatanlou/alephbert-base-finetuned-DSS-maskedLM",
        "yonatanlou/dictabert-finetuned-DSS-maskedLM",
    ],
    "adj_types": {
        "tfidf": {"max_features": 7500},
        "trigram": {"analyzer": "char", "ngram_range": (3, 3)},
        "BOW-n_gram": {"analyzer": "word", "ngram_range": (1, 1)},
        "starr": {},
    },
}


@click.command()
@click.option(
    "--dataset",
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
    default=NUM_COMBINED_GRAPHS,
    help="Number of combined graph types",
)
@click.option("--exp-name", default=EXP_NAME, help="Experiment name")
@click.option(
    "--results-dir", default=GNN_EXP_RESULTS_DIR, help="Directory to store the results"
)
@click.option(
    "--is_supervised",
    is_flag=IS_SUPERVISED,
    help="Run supervised GCN instead of unsupervised GVAE (empty for unsupervised)",
)
@measure_time
def run_gnn_exp_main(
    dataset, domain, num_combined_graphs, exp_name, results_dir, is_supervised
):
    if dataset == "all" and domain == "dss":
        for dataset in DSS_OPTIONAL_DATASETS:
            run_gnn_hyperparameter_tuning(
                dataset,
                domain,
                num_combined_graphs,
                exp_name,
                results_dir,
                is_supervised,
            )
    else:
        if (dataset not in DSS_OPTIONAL_DATASETS) or (
            dataset not in BIBLE_OPTIONAL_DATASETS
        ):
            click.echo(f"Invalid dataset: {dataset}.")
            return


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

    all_param_dicts = generate_parameter_combinations(PARAMS, num_combined_graphs)

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
    vectorizer_types = get_vectorizer_types()
    processor = VectorizerProcessor(
        df, paths["processed_vectorizers_path"], vectorizer_types
    )
    processed_vectorizers = processor.load_or_generate_embeddings()
    df = df.reset_index()

    run_gnn_exp(
        all_param_dicts,
        df,
        processed_vectorizers,
        file_name,
        dataset_info,
        is_supervised,
        verbose=False,
    )


if __name__ == "__main__":
    run_gnn_exp_main()
