import os
from collections import Counter
import numpy as np
import parser_data
import utils
from BERT import bert
from BERT.bert import chars_to_delete
from Starr import starr
from clusters import get_clusters_scores
from constants import TRIGRAM_FEATURE_LENGTH, WORD_PER_SAMPLES
from utils import Transcriptor
import matplotlib.pyplot as plt
import pandas as pd
from os.path import dirname, abspath

np.random.seed(42)

base_dir = dirname(dirname(abspath(__file__)))
CLUSTERS_RESULTS_PATH = f"{base_dir}/Results/Clusters_reconstruction"
section_type = ["sectarian_texts", "straddling_texts", "non_sectarian_texts"]

BOOKS_TO_RUN_ON = [
    "Musar Lamevin",
    "Berakhot",
    "4QM",
    "Catena_Florilegium",
    "4QD",
    "Hodayot",
    "Pesharim",
    "1QH",
    "1QS",
    "Songs_of_Maskil",
    "non_biblical_psalms",
    "Mysteries",
    "4QS",
    "4QH",
    "1QSa",
    "CD",
    "1QM",
    "Collections_of_psalms",
    "Book_of_Jubilees",
]


def main():
    all_scores = []
    book_yml = utils.read_yaml(f"{base_dir}/Data/yamls/all_sectarian_texts.yaml")
    if any(BOOKS_TO_RUN_ON):
        book_dict = {
            k: v
            for d in book_yml.values()
            for k, v in d.items()
            if k in BOOKS_TO_RUN_ON
        }
    else:
        book_dict = {k: v for d in book_yml.values() for k, v in d.items()}
    book_to_section = {b: s for s, d in book_yml.items() for b in d}
    data = parser_data.get_dss_data(book_dict, type="nonbib")
    all_trigram_feature_vector = get_trigram_feature_vectors(data)
    for book_name, book_data in data.items():
        if book_name in os.listdir(f"{CLUSTERS_RESULTS_PATH}"):
            print(f"{book_name} already have results")
            continue
        if not os.path.exists(f"{CLUSTERS_RESULTS_PATH}/{book_name}"):
            os.makedirs(f"{CLUSTERS_RESULTS_PATH}/{book_name}")
        print(f"start parse book: {book_name}")
        section = book_to_section[book_name]
        book_scores = {"text_type": section, "book_name": book_name}
        if len(book_data) < TRIGRAM_FEATURE_LENGTH:
            print(f"{book_name} with size: {len(book_data)}")
            continue

        samples, sample_names = parser_data.get_samples(
            book_data, word_per_samples=WORD_PER_SAMPLES
        )
        if len(samples[-1]) < 50:  # TODO why?
            samples = samples[:-1]
            sample_names = sample_names[:-1]
        print(f'book: {book_scores["book_name"]}')
        bert_features = bert.get_aleph_bert_features(samples, mode_idx=2)
        book_scores = get_clusters_scores(
            bert_features,
            sample_names,
            linkage_criterion="average",
            path=f"{CLUSTERS_RESULTS_PATH}/{book_name}",
            book_scores=book_scores,
            method_name="bert",
            word_per_samples=WORD_PER_SAMPLES,
        )
        starr_features = starr.get_starr_features(samples)
        book_scores = get_clusters_scores(
            starr_features,
            sample_names,
            linkage_criterion="average",
            path=f"{CLUSTERS_RESULTS_PATH}/{book_name}",
            book_scores=book_scores,
            method_name="starr",
            word_per_samples=WORD_PER_SAMPLES,
        )

        trigram_features = np.array(
            [all_trigram_feature_vector[name] for name in sample_names]
        )
        book_scores = get_clusters_scores(
            trigram_features,
            sample_names,
            linkage_criterion="average",
            path=f"{CLUSTERS_RESULTS_PATH}/{book_name}",
            book_scores=book_scores,
            method_name="trigram",
            word_per_samples=WORD_PER_SAMPLES,
        )

        assert bert_features.shape == trigram_features.shape
        bert_matmul_trigram = bert_features @ trigram_features.T
        book_scores = get_clusters_scores(
            bert_matmul_trigram,
            sample_names,
            linkage_criterion="average",
            path=f"{CLUSTERS_RESULTS_PATH}/{book_name}",
            book_scores=book_scores,
            method_name="bert_matmul_trigram",
            word_per_samples=WORD_PER_SAMPLES,
        )
        bert_concat_trigram = np.hstack([trigram_features, bert_features, starr_features])
        print(book_scores["book_name"])
        book_scores = get_clusters_scores(
            bert_concat_trigram,
            sample_names,
            linkage_criterion="average",
            path=f"{CLUSTERS_RESULTS_PATH}/{book_name}",
            book_scores=book_scores,
            method_name="bert_concat_trigram",
            word_per_samples=WORD_PER_SAMPLES,
        )

        all_scores.append(book_scores)

    results = pd.DataFrame(all_scores)
    save_results(results, f"{CLUSTERS_RESULTS_PATH}/scores.csv")


def save_results(results, file_name):
    try:
        old_results = pd.read_csv(file_name)
    except FileNotFoundError:
        old_results = pd.DataFrame()
    if not old_results.empty:
        # Update rows with the same book_name
        old_results = old_results.set_index("book_name")
        results = results.set_index("book_name")
        old_results.update(results)
        old_results = old_results.reset_index()
        results = results.reset_index()

        # Append rows with book_names not in the old scores
        append_rows = results[~results["book_name"].isin(old_results["book_name"])]
        new_results = pd.concat([old_results, append_rows], axis=0)
        new_results.to_csv(file_name, index=False)
    else:
        results.to_csv(file_name, index=False)


def aleph_bert_preprocessing(book_words):
    transcriptor = Transcriptor(f"../Data/yamls/heb_transcript.yaml")
    books_transcript_words = []
    for word in book_words:
        word_list = []
        for entry in word:
            filtered_entry_transcript = chars_to_delete.sub("", entry["transcript"])
            filtered_entry_transcript = filtered_entry_transcript.replace("\xa0", "")
            word_list.append(transcriptor.latin_to_heb(filtered_entry_transcript))
        books_transcript_words.append("".join(word_list))
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
        trigram_samples = [
            Counter([r.replace(".", "")[i : i + 3] for i in range(len(r) - 3)])
            for r in reprocessed_samples
        ]
        # if book_name not in test_books:
        [all_trigram_counter.update(c) for c in trigram_samples]
        trigram_samples = [
            {k: v for k, v in s.items() if k != ""} for s in trigram_samples
        ]
        all_trigram_samples.extend(
            [trigram_samples[i] for i in range(len(sample_names))]
        )
        all_trigram_names.extend([sample_names[i] for i in range(len(sample_names))])

    most_frequent_trigram = sorted(
        [(v, k) for k, v in all_trigram_counter.items() if k.strip() != ""],
        reverse=True,
    )
    most_frequent_trigram = [
        most_frequent_trigram[i][1] for i in range(TRIGRAM_FEATURE_LENGTH)
    ]

    update_trigram_samples = np.array(
        [
            [samples.get(trigram, 0) for trigram in most_frequent_trigram]
            for samples in all_trigram_samples
        ]
    )
    means_trigram = np.mean(update_trigram_samples, axis=0)
    std_trigram = np.std(update_trigram_samples, axis=0)
    normalize_trigram_samples = (update_trigram_samples - means_trigram) / std_trigram
    normalize_trigram_samples = {
        name: normalize_trigram_samples[i] for i, name in enumerate(all_trigram_names)
    }
    return normalize_trigram_samples


def get_bar_graph(feature_names):
    df = pd.read_csv(f"{CLUSTERS_RESULTS_PATH}/scores.csv")

    df = df[df["book_name"] != "Musar Lamevin"]
    scrolls_names = df["book_name"].str.replace("_", "\n").tolist()
    results = {k:[] for k in feature_names}
    significant_dict = {k:[] for k in feature_names}

    for t in feature_names:
        score_col = f"{t}_feature_clusters_score"
        random_mean_col = f"{t}_feature_random_score_mean"
        random_std_col = f"{t}_feature_random_score_std"

        scores = df[score_col].astype(float)
        random_means = df[random_mean_col].astype(float)
        random_stds = df[random_std_col].astype(float)

        significants = (scores - random_means) / random_stds

        results[t] = scores.tolist()
        significant_dict[t] = [round(val, 2) for val in significants]
    results_df = pd.DataFrame(results, index=scrolls_names)


    barWidth = 0.1
    fig, ax = plt.subplots(figsize=(35, 8))

    x_ticks = np.arange(len(scrolls_names))
    bar_dict = {}

    for i, t in enumerate(feature_names):
        x = x_ticks + i * barWidth
        bars = ax.bar(
            x, results_df[t], width=barWidth, edgecolor="grey", label=t, alpha=0.8
        )
        bar_dict[t] = bars

    # Add labels and legends
    ax.set_ylabel("Scores", fontsize=15)
    ax.set_xticks(x_ticks + (len(feature_names) - 1) * barWidth / 2)
    ax.set_xticklabels(scrolls_names, fontsize=8)
    ax.legend()

    for t in feature_names:
        for i, rect in enumerate(bar_dict[t]):
            height = rect.get_height()
            vals = significant_dict[t][i]
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height + 0.003,
                vals,
                ha="center",
                va="bottom",
                fontsize=6,
            )

    plt.savefig(f"{CLUSTERS_RESULTS_PATH}/scores_bars")


main()
get_bar_graph(["bert", "trigram", "starr", "bert_matmul_trigram", "bert_concat_trigram"])
