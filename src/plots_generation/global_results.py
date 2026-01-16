from config import BASE_DIR
from src.plots_generation.plot_utils import (
    get_group_by_vectorizer,
    generate_color_map,
    generate_bar_plot,
    COMPARISON_SCHEMES,
    get_func_by_is_supervised,
    BASE_COLOR_BY_GROUP,
)
import pandas as pd

MAIN_METRICS = {"supervised": "weighted_f1", "unsupervised": "jaccard"}


def clean_vectorizer_names(df: pd.DataFrame, specific_vectorizers=None) -> pd.DataFrame:
    """
    Clean the 'vectorizer' column by keeping only the last part of the path
    and replacing specific names.
    """
    df = df.copy()
    df = df[df["vectorizer"] != "dicta-il/MsBERT"]
    if specific_vectorizers:
        df = df[df["vectorizer"].isin(specific_vectorizers)]


    # if is_supervised:
    #     df = df[df["model"].isin(["MLPClassifier", "GCN"])]

    # segment by vectorizer
    df["vectorizer_type"] = df["vectorizer"].apply(get_group_by_vectorizer)
    df["task"] = df["task"].replace("section", "sectarian")
    df["task"] = df["task"].replace("book", "scroll")
    df["vectorizer"] = df["vectorizer"].str.split("/").str[-1]
    replacements = {
        "BEREL-finetuned-DSS-maskedLM": "BEREL-finetuned",
        "alephbert-base-finetuned-DSS-maskedLM": "AlephBERT-finetuned",
        "alephbert-base": "AlephBERT",
    }
    df["vectorizer"] = df["vectorizer"].replace(replacements)

    return df


def process_data_for_plot(
    domain, is_supervised, gnn_exp_name, gnn_name_format, main_metric, specific_vectorizers=None
):
    # some basic settings

    baseline_dir = f"{BASE_DIR}/experiments/{domain}/cross_validation/baselines"
    baseline_gnn_dir = (
        f"{BASE_DIR}/experiments/{domain}/cross_validation/gnn/{gnn_exp_name}"
    )
    task_by_domain = {"dss": ["book", "composition", "section"], "bible": ["book"]}
    if not is_supervised and domain == "dss":
        task_by_domain["dss"].remove("section")
    tasks = task_by_domain[domain]
    compare_list = {
        task: [
            f"{baseline_dir}/{task}_{'supervised' if is_supervised else 'unsupervised'}.csv",
            f"{baseline_gnn_dir}/{gnn_name_format.format(gnn_exp_name, task)}",
        ]
        for task in tasks
    }
    comparison_scheme = COMPARISON_SCHEMES[
        "supervised" if is_supervised else "unsupervised"
    ]

    # get the data
    all_results = get_func_by_is_supervised(is_supervised)(
        compare_list, tasks, comparison_scheme, main_metric
    )
    all_results = clean_vectorizer_names(all_results, specific_vectorizers)

    return all_results


def make_bar_plot(domain, is_supervised, gnn_exp_name, gnn_name_format, file_name, main_metric=None, specific_vectorizers=None):
    if not main_metric:
        main_metric = MAIN_METRICS["supervised" if is_supervised else "unsupervised"]
    all_results = process_data_for_plot(
        domain, is_supervised, gnn_exp_name, gnn_name_format, main_metric, specific_vectorizers
    )
    color_map = generate_color_map(
        all_results, "vectorizer", "vectorizer_type", "RdYlGn", BASE_COLOR_BY_GROUP
    )

    hue_cols = all_results["vectorizer"].unique()
    plot_obj = generate_bar_plot(
        all_results,
        "model",
        main_metric,
        "vectorizer",
        color_map,
        filename=file_name,
        which_hue_cols=hue_cols,
        base_color_by_group=BASE_COLOR_BY_GROUP,
    )


def make_simple_bar_plot(domain, is_supervised, gnn_exp_name, gnn_name_format, file_name, main_metric=None, specific_vectorizers=None, bar_configs=None):
    """
    Create a bar plot with each bar labeled on the x-axis (no legend grouping).

    bar_configs: list of dicts with keys 'vectorizer', 'model', and optionally 'label'
        e.g., [{'vectorizer': 'tfidf', 'model': 'Only Embeddings', 'label': 'tfidf'},
               {'vectorizer': 'BEREL', 'model': 'GNN', 'label': 'BEREL-GNN'}]
    """
    import scienceplots
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    if not main_metric:
        main_metric = MAIN_METRICS["supervised" if is_supervised else "unsupervised"]

    all_results = process_data_for_plot(
        domain, is_supervised, gnn_exp_name, gnn_name_format, main_metric, specific_vectorizers
    )

    # Filter and prepare data based on bar_configs
    filtered_rows = []
    for config in bar_configs:
        vectorizer = config['vectorizer']
        model = config['model']
        label = config.get('label', vectorizer)

        mask = (all_results['vectorizer'] == vectorizer) & (all_results['model'] == model)
        rows = all_results[mask].copy()
        rows['bar_label'] = label
        filtered_rows.append(rows)

    plot_data = pd.concat(filtered_rows, ignore_index=True)

    # Preserve order from bar_configs
    label_order = [config.get('label', config['vectorizer']) for config in bar_configs]
    plot_data['bar_label'] = pd.Categorical(plot_data['bar_label'], categories=label_order, ordered=True)

    # Generate colors based on vectorizer type
    color_map = generate_color_map(
        plot_data, "vectorizer", "vectorizer_type", "RdYlGn", BASE_COLOR_BY_GROUP
    )
    # Map colors to bar labels
    label_to_vectorizer = {config.get('label', config['vectorizer']): config['vectorizer'] for config in bar_configs}
    bar_colors = [color_map[label_to_vectorizer[label]] for label in label_order]

    plt.style.use(["science", "no-latex"])

    # Create plot for each task
    for task in plot_data["task"].unique():
        print(f"producing for {task=}")
        fig = plt.figure(figsize=(6, 6))
        task_data = plot_data[plot_data["task"] == task]

        sns.barplot(
            x="bar_label",
            y=main_metric,
            data=task_data,
            palette=bar_colors,
            order=label_order
        )

        # Adjust y-axis limits
        min_y = task_data[main_metric].min()
        max_y = task_data[main_metric].max()
        padding = (max_y - min_y) * 0.1
        plt.ylim(min_y - padding, max_y + padding)

        plt.xlabel("", fontsize=16)
        plt.ylabel(main_metric.replace("_", " ").capitalize(), fontsize=14)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=12, rotation=0)
        plt.grid(alpha=0.5)
        plt.tight_layout()

        if file_name:
            task_filename = file_name.format(task)
            if not os.path.exists(os.path.dirname(task_filename)):
                os.makedirs(os.path.dirname(task_filename))
            plt.savefig(task_filename, bbox_inches="tight", dpi=600)
            print(f"Saved plot to {task_filename}")

        plt.show()


if __name__ == "__main__":
    # DOMAINS = ["dss", "bible"]
    DOMAINS = ["dss"]
    # SUPERVISED_OPTIONS = [True, False]
    SUPERVISED_OPTIONS = [False]

    for domain in DOMAINS:
        for is_supervised in SUPERVISED_OPTIONS:
            # Determine gnn_exp_name based on is_supervised
            gnn_exp_name = "gcn_init" if is_supervised else "gae_init"

            # Determine gnn_name_format based on domain and is_supervised
            gnn_name_format = "{}_{}_2_adj_types.csv"

            file_name = (
                f"{BASE_DIR}/reports/plots/global_results/{domain}_{'unsupervised' if not is_supervised else 'supervised'}"
                + "_v2_600dpi_{}.png"
            )

            # Define which bars to show (vectorizer + model combinations)
            bar_configs = [
                {'vectorizer': 'trigram', 'model': 'Only Embeddings', 'label': 'trigram'},
                {'vectorizer': 'tfidf', 'model': 'Only Embeddings', 'label': 'tfidf'},
                {'vectorizer': 'BEREL', 'model': 'Only Embeddings', 'label': 'BEREL'},
                {'vectorizer': 'BEREL', 'model': 'GNN', 'label': 'BEREL-GNN'},
            ]

            make_simple_bar_plot(
                domain, is_supervised, gnn_exp_name, gnn_name_format, file_name, "dasgupta",
                specific_vectorizers=["dicta-il/BEREL", "tfidf", "trigram"],
                bar_configs=bar_configs
            )

