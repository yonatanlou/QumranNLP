import numpy as np
import sklearn.cluster as sk


n_randomizations = 1000


modes = [
    [
        "average",
        4,
        "average_word_vectors",
        "Average of words embeddings of average 4 last hidden layers",
        "average",
    ],
    [
        "concat",
        4,
        "average_word_vectors",
        "Average of words embeddings of concatenation of 4 last hidden layers",
        "average",
    ],
    [
        "last",
        1,
        "average_word_vectors",
        "Average of words embeddings of last hidden layer",
        "average",
    ],
    ["", "", "CLS embedding", "Pooled results of CLS token", "average"],
]


for preset in presets:
    # sentence_vectors = get_sentence_vectors(pretrained_preds, transcripted_samples, wrd_vec_mode=preset[0],
    #                                         wrd_vec_top_n_layers=preset[1], sentence_emb_mode=preset[2],
    #                                         plt_xrange=None, plt_yrange=None, plt_zrange=None,
    #                                         title_prefix="Pretrained model:")
    model = sk.AgglomerativeClustering(
        distance_threshold=0, n_clusters=None, linkage=preset[4]
    )

    random_scores = []
    for i in range(n_randomizations):
        indexes = np.arange(len(transcripted_samples))
        np.random.shuffle(indexes)
        model = sk.AgglomerativeClustering(
            distance_threshold=0, n_clusters=None, linkage=preset[4]
        )
        clusters = model.fit_predict(sentence_vectors[indexes])
        # linkage_matrix = get_linakage_matrix(model)
        # random_scores.append(dasgupta_score(adjacency_matrix, linkage_matrix))
        a = 1
    print(np.mean(random_scores), np.std(random_scores))
