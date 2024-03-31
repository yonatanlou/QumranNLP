from typing import Dict, Any, Tuple

import pandas as pd

from logger import get_logger

logger = get_logger(__name__)


from config import BASE_DIR
from src.features.BERT import bert
from src.hierarchial_clustering.clustering_utils import generate_books_dict

from src.parsers import parser_data
HEBREW_PROCESS_PATH = f"{BASE_DIR}/data/hebrew_processed_files"
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


def parse_text(yaml_book_file, books_to_run, bib_type) -> Tuple[Any, Dict, Dict, str]:
    book_dict, book_to_section = generate_books_dict(books_to_run, yaml_book_file)
    data = parser_data.get_dss_data(book_dict, type=bib_type)
    return (data, book_dict, book_to_section, bib_type)


def add_train_label(all_labels, all_psukim, train_size=0.8):
    sentences_len = len(all_psukim)
    train_size = round(train_size * sentences_len)
    counter = 0
    all_labels_with_train_test_label = []
    for line in all_labels:
        if counter <= train_size:
            all_labels_with_train_test_label.append(f"{line}\ttrain")
        else:
            all_labels_with_train_test_label.append(f"{line}\ttest")
        counter += 1
    return all_labels_with_train_test_label


def write_data_into_text_files(data,book_dict, book_to_section, bib_type, minimum_book_length,train_size):
    all_psukim = []
    all_labels = []
    for book_name, book_data in data.items():
        if len(book_data) < minimum_book_length:
            continue
        samples, sample_names = parser_data.get_samples(book_data, word_per_samples=100)
        if samples is None:
            continue
        preprocessed_samples = bert.aleph_bert_preprocessing(samples)

        for sentence in preprocessed_samples:
            all_psukim.append(sentence)
        for label in sample_names:
            all_labels.append(label)

    df = pd.DataFrame(all_labels)
    df["label"] = df[0].str.split(":").apply(lambda x: x[0])
    book_label_by_count = df['label'].value_counts()
    relevant_books = book_label_by_count[book_label_by_count > 1].index.to_list()
    all_labels_with_train_test_label_book_level_label = []
    relevant_sentences_idx = []
    idx = 0
    for line in all_labels_with_train_test_label:
        book = line.split(':')[0]
        if book in relevant_books:
            all_labels_with_train_test_label_book_level_label.append(f"{line}\t{line.split(':')[0]}")
            relevant_sentences_idx.append(idx)
        idx +=1

    all_labels_with_train_test_label = add_train_label(all_labels, all_psukim, train_size=train_size)
    #TODO rewrite
    all_psukim_filtered = []
    for idx in relevant_sentences_idx:
        all_psukim_filtered.append(all_psukim[idx])


    with open(f"{HEBREW_PROCESS_PATH}/DSS_{bib_type}_labels.txt", "w") as file:
        file.write("\n".join(all_labels_with_train_test_label_book_level_label))
    logger.info(f"{HEBREW_PROCESS_PATH}/DSS_{bib_type}_labels.txt was saved")
    with open(f"{HEBREW_PROCESS_PATH}/DSS_{bib_type}_text.txt", "w") as file:
        file.write("\n".join(all_psukim))
    logger.info(f"{HEBREW_PROCESS_PATH}/DSS_{bib_type}_text.txt was saved")
    with open(f"{HEBREW_PROCESS_PATH}/DSS_{bib_type}_details.txt", "w") as file:
        for k,v in book_dict.items():
            file.write(f"{k}: {v}\n")
        for k,v in book_to_section.items():
            file.write(f"{k}: {v}\n")
    logger.info(f"{HEBREW_PROCESS_PATH}/DSS_{bib_type}_details.txt was saved")






MINIMUM_BOOK_LENGTH = 100
TRAIN_SIZE = 0.8

if __name__ == "__main__":
    (data, book_dict, book_to_section, bib_type) = parse_text("all_sectarian_texts.yaml", BOOKS_TO_RUN_ON, "nonbib")
    write_data_into_text_files(data, book_dict, book_to_section, bib_type, MINIMUM_BOOK_LENGTH, train_size=TRAIN_SIZE)