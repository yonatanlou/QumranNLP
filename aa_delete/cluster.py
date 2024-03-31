from os import path
import sys
from pathlib import Path

from src.parsers import parser_data

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
from sknetwork.hierarchy import dasgupta_score
import sklearn.cluster as sk
from features_keys import feature_list

n_randomizations = 5000


samples, sample_names = zip(
    *[x for x in parser_data.gen_samples(filtered_data["11Q5"], n_words_per_feature)]
)
features = [
    gen_sample_features(sample, morph_name_dict, feature_list) for sample in samples
]
features = np.array(features)


presets = [["average"], ["ward"]]
for preset in presets:
    model = sk.AgglomerativeClustering(
        distance_threshold=0, n_clusters=None, linkage=preset[0]
    )
    clusters = model.fit_predict(features)
    file_name = "Starr Features, {}, {}".format(n_words_per_feature, preset[0])
    title = "Clustering with Morphological Features, {} words per sample, linakage: {}".format(
        n_words_per_feature, preset[0]
    )
    plot_dendrogram(
        model,
        sample_names,
        adjacency_matrix,
        title,
        path.join(output_file_path, file_name),
    )

    random_scores = []
    for i in range(n_randomizations):
        indexes = np.arange(len(features))
        np.random.shuffle(indexes)
        model = sk.AgglomerativeClustering(
            distance_threshold=0, n_clusters=None, linkage=preset[0]
        )
        clusters = model.fit_predict(features[indexes])
        linkage_matrix = get_linakage_matrix(model)
        random_scores.append(dasgupta_score(adjacency_matrix, linkage_matrix))
    print(
        f"Random {preset}: mean: {np.mean(random_scores)}, std: {np.std(random_scores)}"
    )
feature_names = [a[0] for a in feature_list]

# score_table = np.array([scores[book] for book in books_yaml.keys()]).T
# score_table[-1, :] = score_table[-1, :] /100
# feature_df = pd.DataFrame(score_table, index= feature_names,  columns=[book for book in books_yaml.keys()]).round(3)
# feature_df.to_csv('starr_features.csv')
# a=1
