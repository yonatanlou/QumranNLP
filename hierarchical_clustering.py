import os
from collections import Counter
import numpy as np
import parser_data
import utils
from BERT import bert
from BERT.bert import chars_to_delete
from Starr import starr
from clusters import get_clusters_scores
from constants import TRIGRAM_FEATURE_LENGTH
from utils import Transcriptor
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(42)
CLUSTERS_RESULTS_PATH = "Results/Clusters_reconstruction"

CLUSTERS_RESULTS_COLS = ['text_type', 'book_name', 'bert_feature_clusters_score',
                         'bert_feature_random_score_mean', 'bert_feature_random_score_std',
                         'starr_feature_clusters_score', 'starr_feature_random_score_mean',
                         'starr_feature_random_score_std', 'trigram_feature_clusters_score',
                         'trigram_feature_random_score_mean',
                         'trigram_feature_random_score_std']

section_type = ['sectarian_texts', 'straddling_texts', 'non_sectarian_texts']

BOOKS_TO_RUN_ON = ['Musar Lamevin', 'Berakhot', '4QM', 'Catena_Florilegium',
                                                                                # '4QD', 'Hodayot', 'Pesharim', '1QH', '1QS', 'Songs_of_Maskil', 'non_biblical_psalms', 'Mysteries',
                                                                                # '4QS', '4QH', '1QSa', 'CD', 'scores_bars.png', '1QM', 'Collections_of_psalms', 'Book_of_Jubilees'
                                                                                ]
def main():
    all_scores = []
    book_yml = utils.read_yaml("Data/yamls/all_sectarian_texts.yaml")
    if any(BOOKS_TO_RUN_ON):
        book_dict = {k: v for d in book_yml.values() for k, v in d.items() if k in BOOKS_TO_RUN_ON}
    else:
        book_dict = {k: v for d in book_yml.values() for k, v in d.items()}
    book_to_section = {b: s for s, d in book_yml.items() for b in d}
    data = parser_data.get_dss_data(book_dict, type='nonbib')
    all_trigram_feature_vector = get_trigram_feature_vectors(data)
    # data = parser_data.get_dss_data("Data/yamls/all_sectarian_texts.yaml", section=section)
    for book_name, book_data in data.items():
        print(f"start parser book: {book_name}")
        section = book_to_section[book_name]
        book_scores = [section, book_name]
        if len(book_data) < TRIGRAM_FEATURE_LENGTH:
            print(f"{book_name} with size: {len(book_data)}")
            continue
        if book_name in os.listdir(f"{CLUSTERS_RESULTS_PATH}"):
            continue
        if not os.path.exists(f"{CLUSTERS_RESULTS_PATH}/{book_name}"):
            os.makedirs(f"{CLUSTERS_RESULTS_PATH}/{book_name}")
        samples, sample_names = parser_data.get_samples(book_data)
        if len(samples[-1]) < 50: #TODO why?
            samples = samples[:-1]
            sample_names = sample_names[:-1]
        bert_features = bert.get_aleph_bert_features(samples, mode_idx=2)
        score, random_scores_mean, random_scores_std = get_clusters_scores(bert_features, sample_names,
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

        trigram_features = np.array([all_trigram_feature_vector[name] for name in sample_names])
        score, random_scores_mean, random_scores_std = get_clusters_scores(trigram_features, sample_names,
                                                                           linkage_criterion='average',
                                                                           file_name=f"{CLUSTERS_RESULTS_PATH}/{book_name}/Trigram_100_samples_average",
                                                                           title="Trigram, 100 samples, Linkage Criterion: average")
        book_scores.append(score)
        book_scores.append(random_scores_mean)
        book_scores.append(random_scores_std)
        # concat_features = np.hstack([trigram_features, bert_features, starr_features])
        concat_features = bert_features@trigram_features.T
        score, random_scores_mean, random_scores_std = get_clusters_scores(concat_features, sample_names,
                                                                           linkage_criterion='average',
                                                                           file_name=f"{CLUSTERS_RESULTS_PATH}/{book_name}/Concat_features_100_samples_average",
                                                                           title="Concat, 100 samples, Linkage Criterion: average")
        book_scores.append(score)
        book_scores.append(random_scores_mean)
        book_scores.append(random_scores_std)

        all_scores.append(book_scores)

    results = pd.DataFrame(all_scores, columns=CLUSTERS_RESULTS_COLS+['concat_feature_clusters_score',
                         'concat_feature_random_score_mean',
                         'concat_feature_random_score_std'])
    try:
        old_results = pd.read_csv(f"{CLUSTERS_RESULTS_PATH}/scores.csv")
    except FileNotFoundError:
        old_results = pd.DataFrame()

    if not old_results.empty:
        # Update rows with the same book_name
        old_results = old_results.set_index('book_name')
        results = results.set_index('book_name')
        old_results.update(results)
        old_results = old_results.reset_index()

        # Append rows with book_names not in the old scores
        append_rows = results[~results.index.isin(old_results['book_name'])]
        old_results = old_results.append(append_rows).reset_index()
        old_results.to_csv(f'{CLUSTERS_RESULTS_PATH}/scores.csv', index=False)
    else:
        results.to_csv(f'{CLUSTERS_RESULTS_PATH}/scores.csv', index=False)


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
    most_frequent_trigram = [most_frequent_trigram[i][1] for i in range(TRIGRAM_FEATURE_LENGTH)]

    update_trigram_samples = np.array([[samples.get(trigram, 0) for trigram in most_frequent_trigram]
                                       for samples in all_trigram_samples])
    means_trigram = np.mean(update_trigram_samples, axis=0)
    std_trigram = np.std(update_trigram_samples, axis=0)
    normalize_trigram_samples = (update_trigram_samples - means_trigram) / std_trigram
    normalize_trigram_samples = {name: normalize_trigram_samples[i] for i, name in enumerate(all_trigram_names)}
    return normalize_trigram_samples


def get_bar_graph():
    df = pd.read_csv(f'{CLUSTERS_RESULTS_PATH}/scores.csv')

    # Filter out rows with 'book_name' equal to "Musar Lamevin"
    df = df[df['book_name'] != "Musar Lamevin"]

    # Create a list of scrolls_names
    scrolls_names = df['book_name'].str.replace("_", '\n').tolist()
    # Create dictionaries to store the results
    results = {"bert": [], "trigram": [], "starr": [], "concat": []}
    significant_dict = {"bert": [], "trigram": [], "starr": [], "concat": []}

    feature_type = ["bert", "trigram", "starr", "concat"]

    for t in feature_type:
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

    # Create a Pandas DataFrame for the results

    # Set up the bar plot
    barWidth = 0.1
    fig, ax = plt.subplots(figsize=(35, 8))

    x_ticks = np.arange(len(scrolls_names))
    bar_dict = {}

    # Plot the bars for each feature type
    for i, t in enumerate(feature_type):
        x = x_ticks + i * barWidth
        bars = ax.bar(x, results_df[t], width=barWidth, edgecolor='grey', label=t, alpha=0.8)
        bar_dict[t] = bars

    # Add labels and legends
    ax.set_ylabel('Scores', fontsize=15)
    ax.set_xticks(x_ticks + (len(feature_type) - 1) * barWidth / 2)
    ax.set_xticklabels(scrolls_names, fontsize=8)
    ax.legend()

    # Add significant values above the bars
    for t in feature_type:
        for i, rect in enumerate(bar_dict[t]):
            height = rect.get_height()
            vals = significant_dict[t][i]
            ax.text(rect.get_x() + rect.get_width() / 2.0, height + 0.003, vals, ha='center', va='bottom', fontsize=6)

    # Save the plot
    plt.savefig(f'{CLUSTERS_RESULTS_PATH}/scores_bars')



main()
get_bar_graph()

