import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import ttest_ind

from src.baselines.features import shorten_path


def matplotlib_dendrogram(linkage_matrix, sample_names, metadata):
    plt.figure(figsize=(25, 18))
    dendrogram(
        linkage_matrix,
        leaf_label_func=lambda x: shorten_path(sample_names[x]),
        orientation="right",
        color_threshold=0.7 * max(linkage_matrix[:, 2]),  # Adjust color threshold
    )

    plt.title(metadata["vectorizer_type"], fontsize=16)
    plt.ylabel("Scroll:Line-Line")
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )
    plt.tick_params(
        axis="y",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected,
        labelsize=12,
    )
    plt.tight_layout()
    plt.show()


def create_dasgupta_score_plot(
    scores_df,
    chunk_size,
    num_cvs,
    frac,
    linkage_method,
    file_name,
    show_p_val=False,
    p_val_random_test=True,
    show_mean_scores=True,
):
    """
    This function creates a box plot of Dasgupta scores for different vectorizer types.

    Args:
        scores: A dictionary containing the scores.
        chunk_size: Size of the chunks used for vectorization (in words).
        num_cvs: Number of cross-validation folds used.
        frac: Fraction of the data used in each CV sample.
        p_val_random_test: Whether to calculate p-values against a random baseline (True) or own baseline (False).
        show_mean_scores: Whether to display the mean score for each vectorizer group (True).
    """
    mean_rand_score = scores_df["dasgupta_score_rand"].mean()
    long_scores_df = pd.melt(
        scores_df,
        id_vars=["vectorizer"],
        value_vars=["dasgupta_score", "dasgupta_score_rand"],
    )

    p_values = {}
    for vectorizer in scores_df["vectorizer"].unique():
        dasgupta_scores = scores_df[scores_df["vectorizer"] == vectorizer][
            "dasgupta_score"
        ]
        if p_val_random_test:
            rand_scores = scores_df[scores_df["vectorizer"] == vectorizer][
                "dasgupta_score_rand"
            ]
        else:
            rand_scores = scores_df[scores_df["vectorizer"] == vectorizer][
                "dasgupta_score"
            ]
        _, p_value = ttest_ind(dasgupta_scores, rand_scores, alternative="greater")
        p_values[vectorizer] = p_value

    # Set the style of the visualization
    sns.set(style="whitegrid", context="paper", font_scale=1.5)

    # Create the box plot
    plt.figure(figsize=(15, 10))
    ax = sns.boxplot(
        data=long_scores_df, x="vectorizer", y="value", hue="variable", palette="Set3"
    )
    sns.stripplot(
        data=long_scores_df,
        x="vectorizer",
        y="value",
        hue="variable",
        linewidth=1,
        alpha=0.4,
        dodge=True,
        palette="Set3",
    )

    # Plot the mean random score as a dashed line
    plt.axhline(y=mean_rand_score, color="r", linestyle="--", label="Mean random score")

    # Calculate mean for each vectorizer group
    mean_scores = (
        long_scores_df[long_scores_df["variable"] == "dasgupta_score"]
        .groupby("vectorizer")["value"]
        .mean()
    )

    if show_p_val:
        for i, vectorizer in enumerate(p_values.keys()):
            p_val = p_values[vectorizer]
            ax.text(
                i,
                long_scores_df[long_scores_df["vectorizer"] == vectorizer][
                    "value"
                ].min()
                - 0.05,
                f"$p={p_val:.3f}$",
                ha="center",
                va="bottom",
                fontsize=12,
                color="black",
            )

    if show_mean_scores:
        for i, vectorizer in enumerate(long_scores_df["vectorizer"].unique()):
            mean_value = mean_scores.loc[vectorizer]
            ax.text(
                i,
                min(
                    long_scores_df[
                        (long_scores_df["vectorizer"] == vectorizer)
                        & (long_scores_df["variable"] == "dasgupta_score")
                    ]["value"]
                )
                - 0.03,
                f"mean={mean_value:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                color="black",
            )

    # Add a title and labels
    plt.suptitle("Dasgupta Score by Vectorizer Type", fontsize=20)
    plt.title(
        f"Linkage: {linkage_method}, Chunk size: {chunk_size} words, {num_cvs} cross-validation, each CV sample {frac:.0%} from the data",
        fontsize=16,
    )
    plt.xlabel("Vectorizer Type", fontsize=16)
    plt.ylabel("Dasgupta Score", fontsize=16)

    # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(
        handles=handles[0:2],
        labels=labels[0:2],
        title="Score Type",
        fontsize=14,
        title_fontsize=16,
        loc="best",
        ncol=3,
    )

    # Improve layout
    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Remove top and right spines for a cleaner look
    sns.despine(offset=5, trim=True)
    plt.xticks(rotation=45)
    plt.savefig(f"{file_name}.png")
    plt.show()


def create_lca_metric_boxplot(scores_df, file_name):
    # Set the style of the visualization
    sns.set(style="whitegrid", context="paper", font_scale=1.5)

    # Create the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="vectorizer",
        y="max_dist_metric_mean",
        hue="vectorizer",
        data=scores_df,
        palette="Set3",
        legend=False,
    )

    # Add a title and labels
    plt.title("max-dist Metric Mean by Vectorizer Type", fontsize=20)
    plt.xlabel("Vectorizer Type", fontsize=16)
    plt.ylabel("max-dist Metric Mean", fontsize=16)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Improve layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{file_name}.png")
    plt.show()
