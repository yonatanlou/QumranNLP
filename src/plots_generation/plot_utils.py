import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

COMPARISON_SCHEMES = {
    "unsupervised": [
        "task",
        "vectorizer",
        "model",
        "silhouette",
        "clustering_accuracy",
        "jaccard",
        "dasgupta",
        "hidden_dim",
        "adj_type",
    ],
    "supervised": [
        "task",
        "model",
        "vectorizer",
        "test_acc",
        "weighted_f1",
        "micro_f1",
        "macro_f1",
        "adj_type",
    ],
}


def get_group_by_vectorizer(i):
    if "maskedLM" in i:
        return "fine_tuned"
    elif "dicta" in i or "onlplab" in i:
        return "pre_trained_bert"
    else:
        return "classic_text_features"


def generate_color_map(
    df, col, group_name, base_color="PuOr", base_color_by_group=None
):
    groups = df.groupby(group_name)[col].apply(list).to_dict()
    num_groups = df[col].nunique()
    all_colors = list(sns.color_palette(base_color, num_groups))
    color_map = {}
    i = 0
    if not base_color_by_group:
        for group, items in groups.items():
            items = list(set(items))

            for item in items:
                color_map[item] = all_colors[i]
                i += 1
        return color_map

    # Function to generate shades of a base color
    def generate_shades(base_color, items):
        cmap = plt.cm.get_cmap(base_color)
        n_uniuqe_shades = len(set(items))
        # Generate evenly spaced values across the full colormap range
        shade_values = np.linspace(0.4, 0.7, n_uniuqe_shades)
        item_to_shade = {}
        i = 0
        for item in set(items):
            item_to_shade[item] = shade_values[i]
            i += 1

        # Return colors from the full range of the colormap
        return [cmap(item_to_shade[item]) for item in items]

    # Create the color map
    color_map = {}
    for group, items in groups.items():
        shades = generate_shades(base_color_by_group[group], items)
        for item, shade in zip(items, shades):
            color_map[item] = shade
    return color_map


def generate_bar_plot(
    all_results,
    x_col,
    y_col,
    hue_col,
    color_map,
    filename,
    which_hue_cols=False,
    base_color_by_group=None,
):
    import scienceplots

    plt.style.use(["science"])
    # Replace "section" with "sectarian"
    all_results["task"] = all_results["task"].replace("section", "sectarian")

    # Create a custom order based on base_color_by_group
    if base_color_by_group:
        custom_order = []
        for group in base_color_by_group.keys():
            group_items = all_results[all_results["vectorizer_type"] == group][
                hue_col
            ].unique()
            custom_order.extend(group_items)
    else:
        custom_order = None

    # Iterate through unique tasks
    for task in all_results["task"].unique():
        # Create a new figure for each task
        plt.figure(figsize=(3.04 * 3, 4), dpi=200)

        # Filter data for the current task
        task_data = all_results[all_results["task"] == task]
        if which_hue_cols is not None:
            task_data = task_data[task_data[hue_col].isin(which_hue_cols)]

        # Sort the data according to the custom order
        if custom_order:
            task_data[hue_col] = pd.Categorical(
                task_data[hue_col], categories=custom_order, ordered=True
            )
            task_data = task_data.sort_values(hue_col)
        # Create the plot
        sns.barplot(
            x=x_col,
            y=y_col,
            hue=hue_col,
            data=task_data,
            palette=color_map,
            hue_order=custom_order,
        )
        plt.title(
            f"{y_col.replace('_', ' ').capitalize()} by {hue_col.capitalize()} for {task.capitalize()}",
            fontsize=16,
        )

        # Adjusting the y-axis limits
        min_y_col = task_data[y_col].min()
        max_y_col = task_data[y_col].max()
        padding = (max_y_col - min_y_col) * 0.1
        plt.ylim(min_y_col - padding, max_y_col + padding)

        plt.xlabel("Model", fontsize=16)
        plt.ylabel(y_col.replace("_", " ").capitalize(), fontsize=14)
        plt.yticks(fontsize=14)
        # plt.xticks(rotation=45)

        # Customize legend
        plt.legend(
            title="Vectorizer",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=12,
            title_fontsize=14,
        )
        plt.grid(alpha=0.5)
        plt.tight_layout()
        # Save figure with task-specific filename
        if filename:
            task_filename = filename.format(task)
            if not os.path.exists(os.path.dirname(task_filename)):
                os.makedirs(os.path.dirname(task_filename))

            plt.savefig(task_filename, bbox_inches="tight")
            print(f"Saved plot to {task_filename}")

        # plt.show()


def generate_all_results_supervised(
    compare_list, tasks, comparison_scheme, main_metric="val_acc"
):
    results = []
    for task in tasks:
        baseline = pd.read_csv(compare_list[task][0])
        baseline = baseline.rename(columns={"accuracy": "test_acc"})
        baseline = baseline[
            baseline["model"].isin(["LogisticRegression", "MLPClassifier"])
        ]
        baseline = baseline.sort_values(by="test_acc", ascending=False)
        baseline["task"] = task
        baseline["adj_type"] = None
        results.append(baseline[comparison_scheme].to_dict(orient="records"))

        gnn = pd.read_csv(compare_list[task][1])
        gnn = gnn.rename(columns={"bert_model": "vectorizer"})

        max_idx = gnn.groupby("vectorizer")[main_metric].idxmax()
        max_test_acc_rows = gnn.loc[max_idx]
        max_test_acc_rows["model"] = "GCN"
        max_test_acc_rows["task"] = task
        results.append(max_test_acc_rows[comparison_scheme].to_dict(orient="records"))
    all_results = pd.DataFrame([item for sublist in results for item in sublist])
    return all_results


def generate_all_results_unsupervised(
    compare_list, tasks, comparison_scheme, main_metric="jaccard"
):
    results = []
    for task in tasks:
        baseline = pd.read_csv(compare_list[task][0])
        baseline["task"] = task
        baseline["adj_type"] = None
        baseline["hidden_dim"] = None
        baseline["model"] = "Only Embeddings"
        baseline = baseline.rename(columns={"vectorizer_type": "vectorizer"})
        results.append(baseline[comparison_scheme].to_dict(orient="records"))

        gnn = pd.read_csv(compare_list[task][1])
        gnn = gnn.rename(columns={"bert_model": "vectorizer_type"})
        gnn["model"] = "GVAE"
        gnn["task"] = task
        gnn = gnn.rename(columns={"vectorizer_type": "vectorizer"})
        max_idx = gnn.groupby("vectorizer")[main_metric].idxmax()

        max_test_acc_rows = gnn.loc[max_idx]
        max_test_acc_rows["model"] = "GVAE"
        max_test_acc_rows["task"] = task
        results.append(max_test_acc_rows[comparison_scheme].to_dict(orient="records"))

    all_results = pd.DataFrame([item for sublist in results for item in sublist])
    all_results = all_results.sort_values(by=main_metric, ascending=False)
    return all_results


def get_func_by_is_supervised(is_supervised):
    if is_supervised:
        return generate_all_results_supervised
    else:
        return generate_all_results_unsupervised


BASE_COLOR_BY_GROUP = {
    "classic_text_features": "Reds",  # Clear red gradient
    "pre_trained_bert": "Greens",  # Clear green gradient
    "fine_tuned": "Blues",  # Clear blue gradient
}
