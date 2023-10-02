import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from sknetwork.hierarchy import dasgupta_score
import sklearn.cluster as sk


def get_linkage_matrix(model):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix


def plot_dendrogram(model, sample_names, adj_mat, title, fig_name, **kwargs):
    linkage_matrix = get_linkage_matrix(model)

    score = dasgupta_score(adj_mat, linkage_matrix)
    # Plot the corresponding dendrogram
    plt.figure(figsize=(12, 9), dpi=120)
    dendrogram(linkage_matrix, **kwargs, leaf_label_func=lambda x: sample_names[x], orientation='right', )

    plt.title(title, fontsize=16)
    plt.ylabel('Column:Line')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected,
        labelsize=12)

    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()
    return score


def get_clusters_scores(features, sample_names, linkage_criterion, file_name, title):
    model = sk.AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=linkage_criterion)
    model.fit_predict(features)
    adjacency_matrix = np.zeros((len(sample_names), len(sample_names)))
    for i in range(0, adjacency_matrix.shape[0] - 1):
        adjacency_matrix[i, i + 1] = 1
        adjacency_matrix[i + 1, i] = 1

    score = plot_dendrogram(model, sample_names, adjacency_matrix, title, file_name) #TODO Proceed from here
    print(f"clusters scores: {score}")

    random_scores = []
    for i in range(5000):
        indexes = np.arange(len(features))
        np.random.shuffle(indexes)
        model = sk.AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=linkage_criterion)
        model.fit_predict(features[indexes])
        linkage_matrix = get_linkage_matrix(model)
        random_scores.append(dasgupta_score(adjacency_matrix, linkage_matrix))
    print(f"Random {linkage_criterion}: mean: {np.mean(random_scores)}, std: {np.std(random_scores)}")
    return score, np.mean(random_scores), np.std(random_scores)
