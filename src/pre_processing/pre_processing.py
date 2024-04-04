from typing import Dict, Any, Tuple

import pandas as pd

from logger import get_logger


from config import BASE_DIR
from src.features.BERT import bert
from src.hierarchial_clustering.clustering_utils import generate_books_dict
import datetime
import os
from src.parsers import parser_data

HEBREW_PROCESS_PATH = f"{BASE_DIR}/data/hebrew_processed_files"
today = datetime.datetime.now().strftime("%Y-%m-%d")
classify_by = "sectarian"
run_name = f"{today}_{classify_by}_classification"
ckpt_dir = f"{HEBREW_PROCESS_PATH}/{run_name}"
os.makedirs(ckpt_dir, exist_ok=True)

logger = get_logger(__name__, f"{ckpt_dir}/{today}.log")
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


def parse_text(yaml_book_file, books_to_run, bib_type) -> Tuple[Any, Dict, Dict]:
    book_dict, book_to_section = generate_books_dict(books_to_run, yaml_book_file)
    data = parser_data.get_dss_data(book_dict, type=bib_type)
    return (data, book_dict, book_to_section)


def df_conversion(all_psukim, all_labels, book_to_sectarian):
    df = pd.DataFrame({"labels": all_labels, "text": all_psukim})
    df["book"] = df["labels"].str.split(":").str[0]
    df["word_count"] = df["text"].str.split(" ").str.len()
    df = df.merge(book_to_sectarian, on="book")
    book_stats = (
        df.groupby("book")["word_count"].agg(["sum", "count"]).sort_values(by="count")
    )
    book_stats.columns = ["word_count", "lines_count"]
    sectarian_stats = (
        df.groupby("sectarian")["word_count"]
        .agg(["sum", "count"])
        .sort_values(by="count")
    )
    sectarian_stats.columns = ["word_count", "lines_count"]
    logger.info(f"words by book: {book_stats.reset_index().to_dict(orient='records')}")
    logger.info(
        f"words by sectarian: {sectarian_stats.reset_index().to_dict(orient='records')}"
    )
    return df, book_stats


def assign_train_test(group, train_ratio):
    # In edge cases where there are only 2 samples, split them equally
    if len(group) == 2:
        group["train_test"] = ["train", "test"]
    else:
        split_index = int(len(group) * train_ratio)
        group["train_test"] = ["train"] * split_index + ["test"] * (
            len(group) - split_index
        )
    return group


def split_train_test(df, book_stats, train_ratio):
    books_with_enough_data = book_stats[book_stats["lines_count"] > 1].index.to_list()
    df_filtered = df[df["book"].isin(books_with_enough_data)]
    logger.info(f"removed {df.shape[0]-df_filtered.shape[0]} books")
    df_filtered_w_label = (
        df_filtered.groupby("book")
        .apply(assign_train_test, train_ratio)
        .reset_index(drop=True)
    )
    assert df_filtered.shape[0] == df_filtered_w_label.shape[0]
    return df_filtered_w_label


def write_data_into_text_files(
    data, book_dict, book_to_section, bib_type, minimum_book_length, train_ratio, label
):
    if label not in ("sectarian", "book"):
        raise ValueError(f"Please choose either 'sectarian' or 'book'")
    book_to_sectarian = generate_book_to_sectarian(book_dict, book_to_section)
    all_psukim, all_labels = get_labels_and_sentences(data, minimum_book_length)
    df, book_stats = df_conversion(all_psukim, all_labels, book_to_sectarian)
    df_labeled = split_train_test(df, book_stats, train_ratio)

    df_labeled[["labels", "train_test", f"{label}"]].to_csv(
        f"{ckpt_dir}/DSS_{label}_{bib_type}_labels.txt",
        sep="\t",
        index=False,
        header=False,
    )
    logger.info(f"{ckpt_dir}/DSS_{label}_{bib_type}_labels.txt was saved")
    df_labeled["text"].to_csv(
        f"{ckpt_dir}/DSS_{label}_{bib_type}_text.txt", index=False, header=False
    )
    logger.info(f"{ckpt_dir}/DSS_{label}_{bib_type}_text.txt was saved")


def generate_book_to_sectarian(book_dict, book_to_section):
    res = []
    for book, lst_of_chapters in book_dict.items():
        for chapter in lst_of_chapters:
            res.append({"sectarian": book_to_section.get(book), "book": chapter})
    return pd.DataFrame(res)


def get_labels_and_sentences(data, minimum_book_length):
    all_psukim = []
    all_labels = []
    for book_name, book_data in data.items():
        if len(book_data) < minimum_book_length:
            continue
        parsed_sentences, labels = parser_data.get_samples(
            book_data, word_per_samples=WORDS_PER_SAMPLE
        )
        if parsed_sentences is None:
            continue
        hebrew_sentences = bert.aleph_bert_preprocessing(parsed_sentences)

        for sentence in hebrew_sentences:
            all_psukim.append(sentence)
        for label in labels:
            all_labels.append(label)
    logger.info(f"parsed and translate to hebrew: {len(all_psukim)} psukim")
    return all_psukim, all_labels


MINIMUM_BOOK_LENGTH = 100
TRAIN_SIZE = 0.8
BIB_TYPE = "nonbib"
WORDS_PER_SAMPLE = 25  # estimated sentence length

if __name__ == "__main__":
    (data, book_dict, book_to_section) = parse_text(
        "all_sectarian_texts.yaml", BOOKS_TO_RUN_ON, BIB_TYPE
    )
    write_data_into_text_files(
        data,
        book_dict,
        book_to_section,
        BIB_TYPE,
        MINIMUM_BOOK_LENGTH,
        train_ratio=TRAIN_SIZE,
        label="sectarian",
    )
