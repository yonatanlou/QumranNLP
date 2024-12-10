import os

import pandas as pd
from matplotlib import pyplot as plt

from config import BASE_DIR
from src.plots_generation.global_results import process_data_for_plot
from src.plots_generation.plot_utils import (
    generate_color_map,
    BASE_COLOR_BY_GROUP,
)

MAIN_METRICS = {"supervised": "weighted_f1", "unsupervised": "jaccard"}


def make_combined_bar_plot_supervised_unsupervised(domain, file_name):
    # Prepare data for both supervised and unsupervised
    supervised_results = process_data_for_plot(
        domain, True, "gcn_init", "{}_{}_2_adj_types.csv"
    )
    unsupervised_results = process_data_for_plot(
        domain, False, "gvae_init", "{}_{}_1_adj_types.csv"
    )

    # Use the existing generate_bar_plot function with modifications
    def generate_combined_bar_plot(
        supervised_results,
        unsupervised_results,
        x_col,
        hue_col,
        color_map,
        filename,
        which_hue_cols=False,
        base_color_by_group=None,
    ):
        import scienceplots
        import seaborn as sns

        XTICK_FONT_SIZE = 18
        XLAB_FONT_SIZE = 18
        TITLE_FONT_SIZE = 20
        LEGEND_FONT_SIZE = 18
        plt.style.use(["science"])
        y_col_supervised = "weighted_f1"
        y_col_unsupervised = "jaccard"

        # Prepare color order
        if base_color_by_group:
            custom_order = []
            for group in base_color_by_group.keys():
                group_items = supervised_results[
                    supervised_results["vectorizer_type"] == group
                ][hue_col].unique()
                custom_order.extend(group_items)
        else:
            custom_order = None

        # Iterate through unique tasks
        for task in supervised_results["task"].unique():
            # Create a new figure for each task with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.2 * 6, 3.8 * 2), dpi=120)

            # Supervised subplot
            task_data_supervised = supervised_results[
                supervised_results["task"] == task
            ]
            if which_hue_cols is not None:
                task_data_supervised = task_data_supervised[
                    task_data_supervised[hue_col].isin(which_hue_cols)
                ]

            # Sort the data according to the custom order for supervised
            if custom_order:
                task_data_supervised[hue_col] = pd.Categorical(
                    task_data_supervised[hue_col], categories=custom_order, ordered=True
                )
                task_data_supervised = task_data_supervised.sort_values(hue_col)

            # Create supervised plot
            sns.barplot(
                x=x_col,
                y=y_col_supervised,
                hue=hue_col,
                data=task_data_supervised,
                palette=color_map,
                hue_order=custom_order,
                ax=ax1,
                legend=False,
            )
            ax1.set_title(
                f"Supervised {task.capitalize()} classification",
                fontsize=TITLE_FONT_SIZE,
            )
            ax1.set_xlabel("Model", fontsize=XLAB_FONT_SIZE)
            ax1.set_ylabel(y_col_supervised.replace("_", " ").capitalize(), fontsize=14)
            ax1.tick_params(axis="both", labelsize=XTICK_FONT_SIZE)
            ax1.grid(alpha=0.5)
            # Unsupervised subplot
            task_data_unsupervised = unsupervised_results[
                unsupervised_results["task"] == task
            ]
            if which_hue_cols is not None:
                task_data_unsupervised = task_data_unsupervised[
                    task_data_unsupervised[hue_col].isin(which_hue_cols)
                ]

            # Sort the data according to the custom order for unsupervised
            if custom_order:
                task_data_unsupervised[hue_col] = pd.Categorical(
                    task_data_unsupervised[hue_col],
                    categories=custom_order,
                    ordered=True,
                )
                task_data_unsupervised = task_data_unsupervised.sort_values(hue_col)

            # Create unsupervised plot
            sns.barplot(
                x=x_col,
                y=y_col_unsupervised,
                hue=hue_col,
                data=task_data_unsupervised,
                palette=color_map,
                hue_order=custom_order,
                ax=ax2,
            )
            ax2.set_title(
                f"Unsupervised {task.capitalize()} clustering",
                fontsize=TITLE_FONT_SIZE,
            )
            ax2.set_xlabel("Model", fontsize=XLAB_FONT_SIZE)
            ax2.set_ylabel(
                y_col_unsupervised.replace("_", " ").capitalize(), fontsize=14
            )
            ax2.tick_params(axis="both", labelsize=XTICK_FONT_SIZE)
            ax2.grid(alpha=0.5)
            handles, labels = ax2.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",  # Place the legend at the bottom center of the figure
                ncol=3,  # Split the legend into 2 columns
                fontsize=LEGEND_FONT_SIZE,  # Adjust font size to fit the text
                frameon=True,  # Remove the legend frame for cleaner look
            )
            ax2.get_legend().remove()  # Remove the legend from the second plot itself

            # Adjust layout to make space for the legend
            plt.subplots_adjust(bottom=0.27)  # Adjust bottom margin to fit the legend
            # Save figure with task-specific filename
            if filename:
                task_filename = filename.format(task)
                if not os.path.exists(os.path.dirname(task_filename)):
                    os.makedirs(os.path.dirname(task_filename))

                plt.savefig(task_filename, bbox_inches="tight")
                print(f"Saved plot to {task_filename}")
            plt.show()
            plt.close(fig)  # Close the figure to free up memory

    # Generate the combined plot
    color_map = generate_color_map(
        supervised_results,
        "vectorizer",
        "vectorizer_type",
        "RdYlGn",
        BASE_COLOR_BY_GROUP,
    )

    hue_cols = supervised_results["vectorizer"].unique()
    generate_combined_bar_plot(
        supervised_results,
        unsupervised_results,
        x_col="model",
        hue_col="vectorizer",
        color_map=color_map,
        filename=file_name,
        which_hue_cols=hue_cols,
        base_color_by_group=BASE_COLOR_BY_GROUP,
    )


def make_combined_bar_plot_unsupervised(domain, file_name):
    # Prepare data for both supervised and unsupervised
    supervised_results = process_data_for_plot(
        domain, True, "gcn_init", "{}_{}_2_adj_types.csv"
    )
    unsupervised_results = process_data_for_plot(
        domain, False, "gave_init", "{}_{}_2_adj_types.csv"
    )

    # Use the existing generate_bar_plot function with modifications
    def generate_combined_bar_plot(
        unsupervised_results,
        x_col,
        hue_col,
        color_map,
        filename,
        which_hue_cols=False,
        base_color_by_group=None,
    ):
        import scienceplots
        import seaborn as sns

        XTICK_FONT_SIZE = 18
        XLAB_FONT_SIZE = 18
        TITLE_FONT_SIZE = 20
        LEGEND_FONT_SIZE = 18
        YLIM_STRETCH = 0.05
        plt.style.use(["science"])
        y_col_unsupervised_1 = "dasgupta"
        y_col_unsupervised_2 = "jaccard"
        unsupervised_results = unsupervised_results[
            unsupervised_results["vectorizer"] != "starr"
        ]
        unsupervised_results = unsupervised_results[
            ~unsupervised_results["vectorizer"].str.contains("alephbert")
        ]
        # Prepare color order
        if base_color_by_group:
            custom_order = []
            for group in base_color_by_group.keys():
                group_items = unsupervised_results[
                    unsupervised_results["vectorizer_type"] == group
                ][hue_col].unique()
                custom_order.extend(group_items)
        else:
            custom_order = None

        # Iterate through unique tasks
        for task in unsupervised_results["task"].unique():
            # Create a new figure for each task with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.2 * 6, 3.8 * 2), dpi=120)

            # Supervised subplot
            task_data_unsupervised = unsupervised_results[
                unsupervised_results["task"] == task
            ]

            if which_hue_cols is not None:
                task_data_unsupervised = task_data_unsupervised[
                    task_data_unsupervised[hue_col].isin(which_hue_cols)
                ]

            # Sort the data according to the custom order for supervised
            if custom_order:
                task_data_unsupervised[hue_col] = pd.Categorical(
                    task_data_unsupervised[hue_col],
                    categories=custom_order,
                    ordered=True,
                )
                task_data_unsupervised = task_data_unsupervised.sort_values(hue_col)

            # Create supervised plot
            sns.barplot(
                x=x_col,
                y=y_col_unsupervised_1,
                hue=hue_col,
                data=task_data_unsupervised,
                palette=color_map,
                hue_order=custom_order,
                ax=ax1,
                legend=False,
            )
            ax1.set_title(
                f"Unsupervised {task.capitalize()} clustering by {y_col_unsupervised_1.capitalize()}",
                fontsize=TITLE_FONT_SIZE,
            )
            ax1.set_xlabel("Model", fontsize=XLAB_FONT_SIZE)
            ax1.set_ylabel(
                y_col_unsupervised_1.replace("_", " ").capitalize(), fontsize=14
            )
            ax1.set_ylim(
                task_data_unsupervised[y_col_unsupervised_1].min() * (1 - YLIM_STRETCH),
                task_data_unsupervised[y_col_unsupervised_1].max() * (1 + YLIM_STRETCH),
            )
            ax1.tick_params(axis="both", labelsize=XTICK_FONT_SIZE)
            ax1.grid(alpha=0.5)
            # Unsupervised subplot

            # Create unsupervised plot
            sns.barplot(
                x=x_col,
                y=y_col_unsupervised_2,
                hue=hue_col,
                data=task_data_unsupervised,
                palette=color_map,
                hue_order=custom_order,
                ax=ax2,
            )
            ax2.set_title(
                f"Unsupervised {task.capitalize()} clustering by {y_col_unsupervised_2.capitalize()}",
                fontsize=TITLE_FONT_SIZE,
            )
            ax2.set_xlabel("Model", fontsize=XLAB_FONT_SIZE)
            ax2.set_ylabel(
                y_col_unsupervised_2.replace("_", " ").capitalize(), fontsize=14
            )
            ax2.set_ylim(
                task_data_unsupervised[y_col_unsupervised_2].min() * (1 - YLIM_STRETCH),
                task_data_unsupervised[y_col_unsupervised_2].max() * (1 + YLIM_STRETCH),
            )
            ax2.tick_params(axis="both", labelsize=XTICK_FONT_SIZE)
            ax2.grid(alpha=0.5)
            handles, labels = ax2.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",  # Place the legend at the bottom center of the figure
                ncol=3,  # Split the legend into 2 columns
                fontsize=LEGEND_FONT_SIZE,  # Adjust font size to fit the text
                frameon=True,  # Remove the legend frame for cleaner look
            )
            ax2.get_legend().remove()  # Remove the legend from the second plot itself

            # Adjust layout to make space for the legend
            plt.subplots_adjust(bottom=0.27)  # Adjust bottom margin to fit the legend
            # Save figure with task-specific filename
            if filename:
                task_filename = filename.format(task)
                if not os.path.exists(os.path.dirname(task_filename)):
                    os.makedirs(os.path.dirname(task_filename))

                plt.savefig(task_filename, bbox_inches="tight")
                print(f"Saved plot to {task_filename}")
            # plt.show()
            plt.close(fig)  # Close the figure to free up memory

    # Generate the combined plot
    color_map = generate_color_map(
        supervised_results,
        "vectorizer",
        "vectorizer_type",
        "RdYlGn",
        BASE_COLOR_BY_GROUP,
    )

    hue_cols = supervised_results["vectorizer"].unique()
    generate_combined_bar_plot(
        unsupervised_results,
        x_col="model",
        hue_col="vectorizer",
        color_map=color_map,
        filename=file_name,
        which_hue_cols=hue_cols,
        base_color_by_group=BASE_COLOR_BY_GROUP,
    )


make_combined_bar_plot_unsupervised(
    "dss",
    f"{BASE_DIR}/reports/plots/global_results/global_unsupervised_results_2_plt"
    + "_{}.png",
)
