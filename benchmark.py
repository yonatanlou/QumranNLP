import numpy as np
import parser_data
from BERT import bert
from collections import Counter
from sklearn import svm, linear_model
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from hierarchial_clustering.constants import TRIGRAM_FEATURE_LENGTH

section_type = ['non_sectarian_texts', 'sectarian_texts']


def get_trigram_feature_vectors():
    all_trigram_counter = Counter()
    all_trigram_names = []
    all_trigram_samples = []
    labels = []
    for section in section_type:
        book_to_parser = ["1QS"] # "data/yamls/all_sectarian_texts.yaml"
        data = parser_data.get_dss_data(book_to_parser, type='nonbib')
        for book_name, book_data in data.items():
            if len(book_data) < 300:
                print(book_name)
                continue
            samples, sample_names = parser_data.get_samples(book_data, word_per_samples=150)
            reprocessed_samples = bert.aleph_bert_preprocessing(samples)
            trigram_samples = [Counter([r.replace('.', '')[i:i+3] for i in range(len(r) - 3)])
                               for r in reprocessed_samples]
            [all_trigram_counter.update(c) for c in trigram_samples]
            trigram_samples = [{k: v for k, v in s.items() if k != ''} for s in trigram_samples]
            all_trigram_samples.extend([trigram_samples[i] for i in range(len(sample_names))])
            all_trigram_names.extend([sample_names[i] for i in range(len(sample_names))])
            labels.extend([section for _ in range(len(sample_names))])
    most_frequent_trigram = sorted([(v, k) for k, v in all_trigram_counter.items() if k != ''], reverse=True)
    most_frequent_trigram = [most_frequent_trigram[i][1] for i in range(TRIGRAM_FEATURE_LENGTH)]

    update_trigram_samples = np.array([[samples.get(t, 0) for t in most_frequent_trigram]
                                       for samples in all_trigram_samples])
    means_trigram = np.mean(update_trigram_samples, axis=0)
    std_trigram = np.std(update_trigram_samples, axis=0)
    normalize_trigram_samples = (update_trigram_samples - means_trigram) / std_trigram
    return most_frequent_trigram, all_trigram_names, normalize_trigram_samples, np.array(labels)


def calculate_distance(a, b):
    np.sum(np.abs(a - b)) / len(a)


def find_knn(x, train_data, train_labels):
    distances = []
    num_feature = len(x)
    for i, t in enumerate(train_data):
        distances.append((i, np.sum(np.abs(x - t)) / num_feature))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    neighbors = [d[0] for d in distances[:7]]
    neighbors_labels = train_labels[neighbors]

    return distances[:7]


def get_trigram_():
    np.random.seed(42)
    trigram_names, samples_names, trigram_samples, labels = get_trigram_feature_vectors()
    sectarian_samples = [i for i, labels in enumerate(labels) if labels == 'sectarian_texts']
    non_sectarian_samples = [i for i, labels in enumerate(labels) if labels != 'sectarian_texts']
    np.random.shuffle(sectarian_samples)
    np.random.shuffle(non_sectarian_samples)
    train_idx_sectarian = int(len(sectarian_samples) * 0.7)
    train_idx_non_sectarian = int(len(non_sectarian_samples) * 0.7)
    train = sectarian_samples[:train_idx_sectarian] + non_sectarian_samples[:train_idx_non_sectarian]
    test = sectarian_samples[train_idx_sectarian:] + non_sectarian_samples[train_idx_non_sectarian:]
    np.random.shuffle(train)
    np.random.shuffle(test)
    clf = svm.SVC()
    clf.fit(trigram_samples[train], labels[train], sample_weight=[1 if x == 'sectarian_texts' else 8 for x in labels[train]])
    prediction = clf.predict(trigram_samples[test])
    accuracy = np.sum([prediction[i] == labels[test][i] for i in range(len(test))]) / len(prediction)
    print(accuracy)

    pca = PCA(n_components=2)
    X_new = pca.fit_transform(trigram_samples)
    # all_names = list(set([x.split(":")[0] for x in samples_names]))
    plt.scatter([x[0] for i, x in enumerate(X_new) if labels[i] == 'sectarian_texts'],
                [x[1] for i, x in enumerate(X_new) if labels[i] == 'sectarian_texts'],
                label='sectarian_texts', marker="o")
    plt.scatter([x[0] for i, x in enumerate(X_new) if labels[i] != 'sectarian_texts'],
                [x[1] for i, x in enumerate(X_new) if labels[i] != 'sectarian_texts'],
                label='non_sectarian_texts', marker="^")
    plt.legend()
    plt.show()


def get_benchmark_logistic():
    trigram_names, samples_names, trigram_samples, labels = get_trigram_feature_vectors()
    sectarian_samples = [i for i, labels in enumerate(labels) if labels == 'sectarian_texts']
    non_sectarian_samples = [i for i, labels in enumerate(labels) if labels != 'sectarian_texts']
    np.random.shuffle(sectarian_samples)
    np.random.shuffle(non_sectarian_samples)
    train_idx_sectarian = int(len(sectarian_samples) * 0.7)
    train_idx_non_sectarian = int(len(non_sectarian_samples) * 0.7)
    train = sectarian_samples[:train_idx_sectarian] + non_sectarian_samples[:train_idx_non_sectarian]
    test = sectarian_samples[train_idx_sectarian:] + non_sectarian_samples[train_idx_non_sectarian:]
    np.random.shuffle(train)
    np.random.shuffle(test)
    classifier = linear_model.LogisticRegression(class_weight={'sectarian_texts': 1, 'non_sectarian_texts': 7})
    classifier = classifier.fit(trigram_samples[train], labels[train])
    prediction = classifier.predict(trigram_samples[test])
    accuracy = np.sum([prediction[i] == labels[test][i] for i in range(len(test))]) / len(prediction)


def get_bert_feature_vectors():
    embedding = np.load("embedding.npy", allow_pickle=True)
    names, labels = np.load("embedding_labels.npy", allow_pickle=True)
    embedding = np.array(embedding)
    mean = np.mean(embedding, axis=0)
    std = np.std(embedding, axis=0)
    normalized_embedding = (embedding - mean) / std
    sectarian_samples = [i for i, labels in enumerate(labels) if labels == '0']
    non_sectarian_samples = [i for i, labels in enumerate(labels) if labels != '0']
    np.random.shuffle(sectarian_samples)
    np.random.shuffle(non_sectarian_samples)
    train_idx_sectarian = int(len(sectarian_samples) * 0.7)
    train_idx_non_sectarian = int(len(non_sectarian_samples) * 0.7)
    train = sectarian_samples[:train_idx_sectarian] + non_sectarian_samples[:train_idx_non_sectarian]
    test = sectarian_samples[train_idx_sectarian:] + non_sectarian_samples[train_idx_non_sectarian:]
    np.random.shuffle(train)
    np.random.shuffle(test)
    classifier = linear_model.LogisticRegression(class_weight={'0': 1, '1': 6})
    classifier = classifier.fit(normalized_embedding[train], labels[train])
    prediction = classifier.predict(normalized_embedding[test])
    accuracy = np.sum([prediction[i] == labels[test][i] for i in range(len(test))]) / len(prediction)


# get_benchmark_logistic()
# for i in range(100):
#     get_bert_feature_vectors()
get_trigram_()
