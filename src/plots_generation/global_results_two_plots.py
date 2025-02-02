import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots

from config import BASE_DIR
from src.plots_generation.global_results import process_data_for_plot
from src.plots_generation.plot_utils import generate_color_map, BASE_COLOR_BY_GROUP

# Plot styling constants
XTICK_FONT_SIZE = 18
XLAB_FONT_SIZE = 18
TITLE_FONT_SIZE = 20
LEGEND_FONT_SIZE = 18
YLIM_STRETCH = 0.025


def filter_vectorizers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with unwanted vectorizer patterns."""
    pattern = "dictabert|starr|finetuned"
    return df[~df["vectorizer"].str.contains(pattern, regex=True)]


def plot_metric(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    hue_col: str,
    y_col: str,
    color_map: dict,
    custom_order: list = None,
    add_legend: bool = False,
) -> None:
    """
    Plot a bar chart for a specific metric.

    Parameters:
      - ax: The matplotlib axis on which to plot.
      - data: The DataFrame containing the data.
      - x_col: Column for x-axis (e.g. 'model').
      - hue_col: Column used for grouping/hue (e.g. 'vectorizer').
      - y_col: The metric to plot (e.g. 'jaccard' or 'dasgupta').
      - color_map: A dictionary mapping hue values to colors.
      - custom_order: An optional list specifying the order of hue values.
    """
    sns.barplot(
        x=x_col,
        y=y_col,
        hue=hue_col,
        data=data,
        palette=color_map,
        hue_order=custom_order,
        ax=ax,
        legend=True if add_legend else False,
        errorbar=("ci", 90),  # Confidence interval (default is 95)
        err_kws={"linewidth": 1},  # Thicker error bars
        capsize=0.05,  # Adds caps to the error bars
    )
    ax.set_xlabel("Model", fontsize=XLAB_FONT_SIZE)
    ax.set_ylabel(y_col.replace("_", " ").capitalize(), fontsize=XLAB_FONT_SIZE)
    ymin = data[y_col].min() * (1 - YLIM_STRETCH)
    ymax = data[y_col].max() * (1 + YLIM_STRETCH)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis="both", labelsize=XTICK_FONT_SIZE)
    ax.grid(alpha=0.5)


def generate_combined_bar_plot(
    df_metric_1: pd.DataFrame,
    df_metric_2: pd.DataFrame,
    x_col: str,
    hue_col: str,
    color_map: dict,
    filename: str,
    which_hue_cols: list = None,
    base_color_by_group: dict = None,
    metric_1: str = "jaccard",
    metric_2: str = "dasgupta",
) -> None:
    plt.style.use(["science", "no-latex"])

    # Remove unwanted vectorizer entries
    df_metric_1 = filter_vectorizers(df_metric_1)
    df_metric_2 = filter_vectorizers(df_metric_2)

    custom_order = None
    if base_color_by_group:
        custom_order = []
        for group in base_color_by_group:
            group_items = (
                df_metric_1.loc[df_metric_1["vectorizer_type"] == group, hue_col]
                .unique()
                .tolist()
            )
            custom_order.extend(group_items)

    for task in df_metric_1["task"].unique():
        data_metric_1 = df_metric_1[df_metric_1["task"] == task].copy()
        data_metric_2 = df_metric_2[df_metric_2["task"] == task].copy()

        if which_hue_cols is not None:
            data_metric_1 = data_metric_1[data_metric_1[hue_col].isin(which_hue_cols)]
            data_metric_2 = data_metric_2[data_metric_2[hue_col].isin(which_hue_cols)]

        if custom_order:
            data_metric_1[hue_col] = pd.Categorical(
                data_metric_1[hue_col], categories=custom_order, ordered=True
            )
            data_metric_1.sort_values(hue_col, inplace=True)
            data_metric_2[hue_col] = pd.Categorical(
                data_metric_2[hue_col], categories=custom_order, ordered=True
            )
            data_metric_2.sort_values(hue_col, inplace=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.2, 7.6), dpi=120)

        plot_metric(
            ax1,
            data_metric_1,
            x_col,
            hue_col,
            metric_1,
            color_map,
            custom_order,
            add_legend=False,
        )
        plot_metric(
            ax2,
            data_metric_2,
            x_col,
            hue_col,
            metric_2,
            color_map,
            custom_order,
            add_legend=True,
        )

        # Add shared legend below the plots using handles from ax2
        # Extract the legend from ax2 and create a shared figure legend.
        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.09),
            fontsize=LEGEND_FONT_SIZE,
            frameon=True,
        )
        # Remove the legend from ax2 so it only appears once.
        leg = ax2.get_legend()
        if leg is not None:
            leg.remove()

        plt.subplots_adjust(bottom=0.27)
        if filename:
            task_filename = filename.format(task)
            os.makedirs(os.path.dirname(task_filename), exist_ok=True)
            plt.savefig(task_filename, bbox_inches="tight")
            print(f"Saved plot to {task_filename}")
        plt.close(fig)


def make_combined_bar_plot_unsupervised(
    domain: str, metric_1, metric_2, file_name: str
) -> None:
    """
    Process data and generate combined bar plots for unsupervised results.

    The function processes data for both 'jaccard' and 'dasgupta' metrics,
    cleans up vectorizer names, generates a color map, and then creates the plots.
    """
    df_metric_1 = process_data_for_plot(
        domain, False, "gae_init", "{}_{}_2_adj_types.csv", main_metric=metric_1
    )
    df_metric_2 = process_data_for_plot(
        domain, False, "gae_init", "{}_{}_2_adj_types.csv", main_metric=metric_2
    )

    # Generate a color map for the plots (same vectorizers so the same color map)
    color_map = generate_color_map(
        df_metric_1, "vectorizer", "vectorizer_type", "RdYlGn", BASE_COLOR_BY_GROUP
    )
    hue_cols = df_metric_1["vectorizer"].unique()

    generate_combined_bar_plot(
        df_metric_1,
        df_metric_2,
        x_col="model",
        hue_col="vectorizer",
        color_map=color_map,
        filename=file_name,
        which_hue_cols=hue_cols,
        base_color_by_group=BASE_COLOR_BY_GROUP,
        metric_1=metric_1,
        metric_2=metric_2,
    )


if __name__ == "__main__":
    make_combined_bar_plot_unsupervised(
        "dss",
        "jaccard",
        "dasgupta",
        f"{BASE_DIR}/reports/plots/global_results/global_unsupervised_gae_results_2_plt_{{}}_.pdf",
    )
