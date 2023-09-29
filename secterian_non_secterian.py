import os
from collections import Counter
import numpy as np
import parser_data
import utils
from BERT import bert
from BERT.bert import chars_to_delete
from Starr import starr
from clusters import get_clusters_scores
from utils import Transcriptor
import csv
import matplotlib.pyplot as plt
CLUSTERS_RESULTS_PATH = "Results/Clusters_reconstruction"

section_type = ['sectarian_texts', 'straddling_texts', 'non_sectarian_texts']


def main():
    all_scores = []
    book_yml = utils.read_yaml("Data/yamls/all_sectarian_texts.yaml")
    book_dict = {k: v for d in book_yml.values() for k, v in d.items()}
    book_to_section = {b: s for s, d in book_yml.items() for b in d}
    data = parser_data.get_dss_data(book_dict, type='nonbib')
    all_trigram_feature_vector = get_trigram_feature_vectors(data)
    # data = parser_data.get_dss_data("Data/yamls/all_sectarian_texts.yaml", section=section)
    for book_name, book_data in data.items():
        print(f"start parser book: {book_name}")
        section = book_to_section[book_name]
        book_scores = [section, book_name]
        if len(book_data) < 500:
            print(f"{book_name} with size: {len(book_data)}")
            continue
        if book_name in os.listdir(f"{CLUSTERS_RESULTS_PATH}"):
            continue
        if not os.path.exists(f"{CLUSTERS_RESULTS_PATH}/{book_name}"):
            os.makedirs(f"{CLUSTERS_RESULTS_PATH}/{book_name}")
        samples, sample_names = parser_data.get_samples(book_data)
        if len(samples[-1]) < 50:
            samples = samples[:-1]
            sample_names = sample_names[:-1]
        bert_feature = bert.get_aleph_bert_features(samples, mode_idx=2)
        score, random_scores_mean, random_scores_std = get_clusters_scores(bert_feature, sample_names,
                                                                           linkage_criterion='average',
                                                                           file_name=f"{CLUSTERS_RESULTS_PATH}/{book_name}/bert_100_samples_average",
                                                                           title="Aleph Bert, 100 samples, Linkage Criterion: average")
        book_scores.append(score)
        book_scores.append(random_scores_mean)
        book_scores.append(random_scores_std)

        starr_features = starr.get_starr_features(samples)
        score, random_scores_mean, random_scores_std = get_clusters_scores(starr_features, sample_names,
                                                                           linkage_criterion='average',
                                                                           file_name=f"{CLUSTERS_RESULTS_PATH}/{book_name}/starr_100_samples_average",
                                                                           title="Starr, 100 samples, Linkage Criterion: average")
        book_scores.append(score)
        book_scores.append(random_scores_mean)
        book_scores.append(random_scores_std)

        trigram_feature = np.array([all_trigram_feature_vector[name] for name in sample_names])
        score, random_scores_mean, random_scores_std = get_clusters_scores(trigram_feature, sample_names,
                                                                           linkage_criterion='average',
                                                                           file_name=f"{CLUSTERS_RESULTS_PATH}/{book_name}/Trigram_100_samples_average",
                                                                           title="Trigram, 100 samples, Linkage Criterion: average")
        book_scores.append(score)
        book_scores.append(random_scores_mean)
        book_scores.append(random_scores_std)

        all_scores.append(book_scores)

    # for section in section_type:
    #

    with open('{CLUSTERS_RESULTS_PATH}/scores.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text_type', 'book_name', 'bert_feature_clusters_score',
                         'bert_feature_random_score_mean', 'bert_feature_random_score_std',
                         'starr_feature_clusters_score', 'starr_feature_random_score_mean',
                         'starr_feature_random_score_std', 'trigram_feature_clusters_score',
                         'trigram_feature_random_score_mean', 'trigram_feature_random_score_std'])
        writer.writerows(all_scores)


def aleph_bert_preprocessing(book_words):
    transcriptor = Transcriptor(f"Data/yamls/heb_transcript.yaml")
    books_transcript_words = []
    for word in book_words:
        word_list = []
        for entry in word:
            filtered_entry_transcript = chars_to_delete.sub('', entry['transcript'])
            filtered_entry_transcript = filtered_entry_transcript.replace(u'\xa0', u'')
            word_list.append(transcriptor.latin_to_heb(filtered_entry_transcript))
        books_transcript_words.append(''.join(word_list))
    return books_transcript_words


def get_trigram_feature_vectors(data):
    all_trigram_counter = Counter()
    all_trigram_names = []
    all_trigram_samples = []
    for book_name, book_data in data.items():
        if len(book_data) < 100:
            continue
        samples, sample_names = parser_data.get_samples(book_data, word_per_samples=100)
        if samples is None:
            continue
        reprocessed_samples = bert.aleph_bert_preprocessing(samples)
        trigram_samples = [Counter([r.replace('.', '')[i:i+3] for i in range(len(r) - 3)]) for r in reprocessed_samples]
        # if book_name not in test_books:
        [all_trigram_counter.update(c) for c in trigram_samples]
        trigram_samples = [{k: v for k, v in s.items() if k != ''} for s in trigram_samples]
        all_trigram_samples.extend([trigram_samples[i] for i in range(len(sample_names))])
        all_trigram_names.extend([sample_names[i] for i in range(len(sample_names))])

    most_frequent_trigram = sorted([(v, k) for k, v in all_trigram_counter.items() if k.strip() != ''], reverse=True)
    most_frequent_trigram = [most_frequent_trigram[i][1] for i in range(500)]

    update_trigram_samples = np.array([[samples.get(trigram, 0) for trigram in most_frequent_trigram]
                                       for samples in all_trigram_samples])
    means_trigram = np.mean(update_trigram_samples, axis=0)
    std_trigram = np.std(update_trigram_samples, axis=0)
    normalize_trigram_samples = (update_trigram_samples - means_trigram) / std_trigram
    normalize_trigram_samples = {name: normalize_trigram_samples[i] for i, name in enumerate(all_trigram_names)}
    return normalize_trigram_samples


def get_bar_graph():
    feature_type = ["bert", "trigram", "starr"]
    results = {"bert": [], "trigram": [], "starr": []}
    scrolls_names = []
    significant_dict = {"bert": [], "trigram": [], "starr": []}

    with open(f'{CLUSTERS_RESULTS_PATH}/scores.csv', newline='') as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if raw['book_name'] == "Musar Lamevin":
                continue
            scrolls_names.append(raw['book_name'].replace("_", '\n'))
            for t in feature_type:
                score = float(raw[f"{t}_feature_clusters_score"])
                random_mean = float(raw[f"{t}_feature_random_score_mean"])
                random_std = float(raw[f"{t}_feature_random_score_std"])
                significant = float((score - random_mean) / random_std)
                results[t].append(score)
                significant_dict[t].append(round(significant, 2))

    barWidth = 0.25
    fig = plt.subplots(figsize=(35, 8))
    bias = 0
    x_ticks = {}
    for t in feature_type:
        x_ticks[t] = [x + bias for x in np.arange(len(scrolls_names))]
        bias += barWidth

    bar_dict = {}
    for k, x in x_ticks.items():
        bar_dict[k] = plt.bar(x_ticks[k], results[k], width=barWidth, edgecolor='grey', label=k, alpha=0.8)

    plt.ylabel('Scores', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(scrolls_names))], scrolls_names, fontsize=8)
    plt.legend()

    for t in feature_type:
        for i, rect in enumerate(bar_dict[t]):
            height = rect.get_height()
            vals = significant_dict[t][i]
            plt.text(rect.get_x() + rect.get_width() / 2.0, height + 0.003, vals, ha='center', va='bottom', fontsize=6)

    plt.savefig(f'{CLUSTERS_RESULTS_PATH}/scores_bars')


# get_bar_graph()

main()