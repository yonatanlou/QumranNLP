import os
import pickle
import click
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from base_utils import measure_time
from config import BASE_DIR, get_paths_by_domain
from src.baselines.utils import create_adjacency_matrix, set_seed_globally
from src.baselines.create_datasets import (
    QumranDataset,
    save_dataset_for_finetuning,
    create_dss_datasets,
)
from src.baselines.embeddings import get_vectorizer_types, VectorizerProcessor
from src.baselines.ml import evaluate_unsupervised_metrics, evaluate_supervised_metrics

MODELS = [
    LogisticRegression(max_iter=500),
    LinearSVC(),
    KNeighborsClassifier(),
    MLPClassifier(random_state=42, max_iter=500),
]


@measure_time
def make_baselines_results(
    domain,
    results_dir,
    train_frac,
    val_frac,
    tasks=["scroll", "composition", "sectarian"],
):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    paths = get_paths_by_domain(domain)

    df_origin = pd.read_csv(paths["data_csv_path"])

    vectorizers = get_vectorizer_types(domain)
    processor = VectorizerProcessor(
        df_origin, paths["processed_vectorizers_path"], vectorizers
    )
    processed_vectorizers = processor.load_or_generate_embeddings()
    set_seed_globally()
    adjacency_matrix_all = create_adjacency_matrix(
        df_origin,
        context_similiarity_window=3,
        composition_level=False,
    )

    datasets, _ = create_dss_datasets(
        tasks,
        train_frac,
        val_frac,
        paths,
        vectorizers,
        df_frac_remove=0,
        seed=42,
        save_dataset=True,
    )
    for dataset_name, dataset in datasets.items():
        set_seed_globally()
        print(f"calculating metrics for {dataset_name}")
        evaluate_unsupervised_metrics(
            adjacency_matrix_all, dataset, vectorizers, results_dir
        )
        evaluate_supervised_metrics(MODELS, vectorizers, dataset, results_dir)


@click.command()
@click.option(
    "--domain",
    type=click.Choice(["dss", "bible"], case_sensitive=False),
    multiple=False,
    default="dss",
    help="dss or bible",
)
@click.option(
    "--results-dir",
    type=click.Path(),
    default=f"{BASE_DIR}/experiments/dss/baselines",
    help="Directory to store results",
)
@click.option(
    "--train-frac", type=float, default=0.7, help="Fraction of data to use for training"
)
@click.option(
    "--val-frac", type=float, default=0.1, help="Fraction of data to use for validation"
)
def run_baselines(domain, results_dir, train_frac, val_frac):
    """Run baseline experiments for the Qumran dataset with specified parameters."""
    click.echo(f"Running baselines with the following configuration:")
    click.echo(f"Domain: {domain}")
    click.echo(f"Results Directory: {results_dir}")
    click.echo(f"Train Fraction: {train_frac}")

    make_baselines_results(
        domain=domain,
        results_dir=results_dir,
        train_frac=train_frac,
        val_frac=val_frac,
    )


if __name__ == "__main__":
    run_baselines()

# python src/baselines/main.py --domain dss --results-dir QumranNLP/experiments/baselines --train-frac 0.7 --val-frac 0.1 --tasks scroll composition sectarian
