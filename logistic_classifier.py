import numpy as np
from collections import Counter, defaultdict

from tqdm import tqdm
from sklearn.decomposition import PCA
from src.parsers import parser_data
from src.features.BERT import bert
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

from config import BASE_DIR
from src.hierarchial_clustering.constants import TRIGRAM_FEATURE_LENGTH

section_type = ["non_sectarian_texts", "sectarian_texts"]


def get_bert_feature_vectors():
    embedding = np.load(
        f"{BASE_DIR}/src/features/BERT/embedding.npy", allow_pickle=True
    )
    names, labels = np.load(
        f"{BASE_DIR}/src/features/BERT/embedding_labels.npy", allow_pickle=True
    )
    mean = np.mean(embedding, axis=0)
    std = np.std(embedding, axis=0)
    normalized_embedding = (embedding - mean) / std
    return normalized_embedding, names, labels


def get_trigram_feature_vectors(test_books):
    all_trigram_counter = Counter()
    all_trigram_names = []
    all_trigram_samples = []
    labels = []
    for section in ["bib", "nonbib"]:
        data = parser_data.get_dss_data(books_list=test_books, type=section)
        for book_name, book_data in data.items():
            if len(book_data) < 100:
                continue
            samples, sample_names = parser_data.get_samples(
                book_data, word_per_samples=100
            )
            if samples is None:
                continue
            reprocessed_samples = bert.aleph_bert_preprocessing(samples)
            trigram_samples = [
                Counter([r.replace(".", "")[i : i + 3] for i in range(len(r) - 3)])
                for r in reprocessed_samples
            ]
            if book_name not in test_books:
                [all_trigram_counter.update(c) for c in trigram_samples]
            trigram_samples = [
                {k: v for k, v in s.items() if k != ""} for s in trigram_samples
            ]
            all_trigram_samples.extend(
                [trigram_samples[i] for i in range(len(sample_names))]
            )
            all_trigram_names.extend(
                [sample_names[i] for i in range(len(sample_names))]
            )
            labels.extend([section for _ in range(len(sample_names))])

    most_frequent_trigram = sorted(
        [(v, k) for k, v in all_trigram_counter.items() if k != ""], reverse=True
    )
    most_frequent_trigram = [
        most_frequent_trigram[i][1] for i in range(TRIGRAM_FEATURE_LENGTH)
    ]

    update_trigram_samples = np.array(
        [
            [samples.get(t, 0) for t in most_frequent_trigram]
            for samples in all_trigram_samples
        ]
    )
    means_trigram = np.mean(update_trigram_samples, axis=0)
    std_trigram = np.std(update_trigram_samples, axis=0)
    normalize_trigram_samples = (update_trigram_samples - means_trigram) / std_trigram
    return normalize_trigram_samples, all_trigram_names, np.array(labels)


def logistic_regressions(
    samples,
    sample_names,
    labels,
    sec_test_books,
    non_sec_test_books,
    class_weight,
    sample_weight,
):
    name = [x.split(":")[0] for x in sample_names]
    # test_books = sec_test_books + non_sec_test_books
    train = [
        i
        for i in range(len(samples))
        if name[i] not in (sec_test_books + non_sec_test_books)
    ]
    test = [
        i
        for i in range(len(samples))
        if name[i] in (sec_test_books + non_sec_test_books)
    ]

    np.random.shuffle(train)
    np.random.shuffle(test)
    classifier = linear_model.LogisticRegression(
        max_iter=300, class_weight=class_weight
    )
    classifier = classifier.fit(
        samples[train], labels[train], sample_weight=sample_weight
    )
    roc_auc = roc_auc_score(
        labels[test],
        classifier.predict_proba(samples[test])[:, 1],
        multi_class="ovr",
        labels=["sectarian_texts", "non_sectarian_texts"],
    )
    prediction = classifier.predict(samples[test])
    accuracy = np.sum(
        [prediction[i] == labels[test[i]] for i in range(len(test))]
    ) / len(test)
    true_positive = np.sum(
        [
            1
            for i in range(len(test))
            if prediction[i] == "sectarian_texts"
            and labels[test][i] == "sectarian_texts"
        ]
    )
    false_positive = np.sum(
        [
            1
            for i in range(len(test))
            if prediction[i] == "sectarian_texts"
            and labels[test][i] == "non_sectarian_texts"
        ]
    )
    true_negative = np.sum(
        [
            1
            for i in range(len(test))
            if prediction[i] == "non_sectarian_texts"
            and labels[test][i] == "non_sectarian_texts"
        ]
    )
    false_negative = np.sum(
        [
            1
            for i in range(len(test))
            if prediction[i] == "non_sectarian_texts"
            and labels[test][i] == "sectarian_texts"
        ]
    )
    # print(f"true_positive: {true_positive}, false_positive: {false_positive}, true_negative: {true_negative}, "
    #       f"false_negative: {false_negative}, accuracy: {accuracy}")

    # print(f"roc: {roc_auc}, acc sec: {accuracy_sec}, acc non sec: {accuracy_non_sec}, "
    #       f"sectarian:{len([t for t in test if name[t] in sec_test_books])}, "
    #       f"non sectarian:{len([t for t in test if name[t] in non_sec_test_books])}")
    return accuracy, roc_auc


def run_experiment(number_expr):
    bert_samples, bert_sample_names, bert_labels = get_bert_feature_vectors()
    sectarian_dict = defaultdict(int)
    non_sectarian_dict = defaultdict(int)
    for i in range(len(bert_sample_names)):
        name = bert_sample_names[i].split(":")[0]
        if bert_labels[i] == "sectarian_texts":
            sectarian_dict[name] += 1
        elif bert_labels[i] == "non_sectarian_texts":
            non_sectarian_dict[name] += 1
        else:
            print(name, bert_labels[i])

    sectarian_books = [k for k, val in sectarian_dict.items() if val > 2 and val < 12]
    non_sectarian_books = [
        k for k, val in non_sectarian_dict.items() if val > 4 and val < 12
    ]
    res_benchmark_acc = []
    res_benchmark_roc = []
    res_bert_acc = []
    res_bert_roc = []

    for j in tqdm(range(number_expr)):
        np.random.shuffle(sectarian_books)
        np.random.shuffle(non_sectarian_books)
        test_books = sectarian_books[:4] + non_sectarian_books[:2]
        (
            trigram_samples,
            trigram_sample_names,
            trigram_labels,
        ) = get_trigram_feature_vectors(test_books)
        name = [x.split(":")[0] for x in trigram_sample_names]
        book_counter = Counter(name)
        train_category = [
            "sectarian_text" if name[i] in sectarian_dict else "non_sectarian_text"
            for i in range(len(trigram_sample_names))
        ]
        type_counter = Counter(train_category)
        # type_weight = type_counter['sectarian_text'] / type_counter['non_sectarian_text']
        samples_weight = [1 / book_counter[n] for n in name if n not in test_books]
        class_weight = {
            "sectarian_texts": 1,
            "non_sectarian_texts": type_counter["sectarian_text"]
            / type_counter["non_sectarian_text"],
        }
        # print(f"sectarian_book:{str(non_sectarian_books[:4])}, non_sectarian_book:{str(non_sectarian_books[:2])}")
        # print("Benchmark")
        accuracy, roc_auc = logistic_regressions(
            samples=trigram_samples,
            sample_names=trigram_sample_names,
            labels=trigram_labels,
            sec_test_books=sectarian_books[:4],
            non_sec_test_books=non_sectarian_books[:2],
            class_weight=class_weight,
            sample_weight=samples_weight,
        )

        res_benchmark_acc.append(accuracy)
        res_benchmark_roc.append(roc_auc)
        # print("Bert")
        name = [x.split(":")[0] for x in bert_sample_names]
        book_counter = Counter(name)
        samples_weight = [1 / book_counter[n] for n in name if n not in test_books]
        accuracy, roc_auc = logistic_regressions(
            samples=bert_samples,
            sample_names=bert_sample_names,
            labels=bert_labels,
            sec_test_books=sectarian_books[:4],
            non_sec_test_books=non_sectarian_books[:2],
            class_weight=class_weight,
            sample_weight=samples_weight,
        )

        res_bert_acc.append(accuracy)
        res_bert_roc.append(roc_auc)
    print(
        f"benchmark acc - mean: {np.mean(res_benchmark_acc)}, std: {np.std(res_benchmark_acc)}, min: {np.min(res_benchmark_acc)}"
    )
    print(
        f"benchmark roc - mean: {np.mean(res_benchmark_roc)}, std: {np.std(res_benchmark_roc)}, min: {np.min(res_benchmark_roc)}"
    )
    print(
        f"bert acc - mean: {np.mean(res_bert_acc)}, std: {np.std(res_bert_acc)}, min: {np.min(res_bert_acc)}"
    )
    print(
        f"bert roc - mean: {np.mean(res_bert_roc)}, std: {np.std(res_bert_roc)}, min: {np.min(res_bert_roc)}"
    )
    # print(f"bert - mean: {np.mean(res_bert)}, std: {np.std(res_bert)}")


def get_pca():
    trigram_samples, samples_names, labels = get_bert_feature_vectors()
    pca = PCA(n_components=2)
    X_new = pca.fit_transform(trigram_samples)
    all_names = list(set([x.split(":")[0] for x in samples_names]))
    plt.scatter(
        [x[0] for i, x in enumerate(X_new) if labels[i] == "sectarian_texts"],
        [x[1] for i, x in enumerate(X_new) if labels[i] == "sectarian_texts"],
        label="sectarian_texts",
        marker="o",
    )
    plt.scatter(
        [x[0] for i, x in enumerate(X_new) if labels[i] != "sectarian_texts"],
        [x[1] for i, x in enumerate(X_new) if labels[i] != "sectarian_texts"],
        label="non_sectarian_texts",
        marker="^",
    )
    plt.legend()
    plt.show()


# get_pca()
run_experiment(number_expr=100)
