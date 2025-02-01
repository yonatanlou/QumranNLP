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

def clean_vectorizer_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the 'vectorizer' column by keeping only the last part of the path
    and replacing specific names.
    """
    df = df.copy()
    df = df[df["vectorizer"] != "dicta-il/MsBERT"]
    # if is_supervised:
    #     df = df[df["model"].isin(["MLPClassifier", "GCN"])]

    # segment by vectorizer
    df["vectorizer_type"] = df["vectorizer"].apply(
        get_group_by_vectorizer
    )
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
    domain, is_supervised, gnn_exp_name, gnn_name_format, main_metric
):
    # some basic settings

    baseline_dir = f"{BASE_DIR}/experiments/{domain}/baselines"
    baseline_gnn_dir = f"{BASE_DIR}/experiments/{domain}/gnn/{gnn_exp_name}"
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
    all_results = clean_vectorizer_names(all_results)


    return all_results


def make_bar_plot(domain, is_supervised, gnn_exp_name, gnn_name_format, file_name):
    main_metric = MAIN_METRICS["supervised" if is_supervised else "unsupervised"]
    all_results = process_data_for_plot(
        domain, is_supervised, gnn_exp_name, gnn_name_format
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
                + "_{}.png"
            )
            make_bar_plot(
                domain, is_supervised, gnn_exp_name, gnn_name_format, file_name
            )
