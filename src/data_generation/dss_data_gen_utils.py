import re
import logger


import pandas as pd
import yaml
from tqdm.auto import tqdm

from config import BASE_DIR
from src.data_generation.constants import (
    NOT_HEB_BOOKS,
    ALLOWED_CHARS,
    manually_remove_scrolls,
    MIN_WORDS_PER_SCROLL,
)
from src.features.Starr import starr
from src.parsers import parser_data
from logger import get_logger

logger = get_logger(__name__)


def remove_not_heb_chars(word):
    new_word = []
    for char in word:
        if char in ALLOWED_CHARS:
            new_word.append(char)
    return "".join(new_word)


def clean_text(text_origin):
    """Clean the text by removing specific Hebrew punctuation and extra spaces."""
    text = text_origin.replace("׃", "").replace("׳", "")
    text = remove_hebrew_punctuation(text)
    text = remove_not_heb_chars(text)
    text = re.sub(" +", " ", text)  # Remove extra spaces

    text = text.replace("\xa0", "")  # Remove non-breaking spaces
    return text


def process_entry(entry, text_field, lex):
    """Process an individual entry based on the specified text field and lexeme flag."""
    if lex and entry["parsed_morph"]["sp"] == "ptcl":
        return ""  # Skip particles when working with lexemes (e.g., ו, ב, etc.)
    return entry[text_field] + " " if lex else entry[text_field]


def process_sample(sample, text_field, lex):
    """Convert a single sample to text, cleaning it and removing unnecessary characters."""
    sample_text = "".join(
        process_entry(entry, text_field, lex) for entry in sample if entry[text_field]
    )
    return clean_text(sample_text)


def process_sample(sample, text_field, lex):
    """Convert a single sample to text, cleaning it and removing unnecessary characters."""
    sample_text = ""
    for entry in sample:
        if entry[text_field]:
            sample_text += process_entry(entry, text_field, lex)
    text = clean_text(sample_text)
    return text


def samples_to_text(samples, lex=False):
    """
    Convert samples to text. If lex is True, use lexeme field; otherwise, use transcript field.

    Parameters:
    samples (list): List of samples to be processed.
    lex (bool): Flag indicating whether to use lexemes instead of transcripts.

    Returns:
    list: List of processed Hebrew samples as text.
    """
    text_field = "lex" if lex else "transcript"
    return [process_sample(sample, text_field, lex) for sample in samples]


def remove_hebrew_punctuation(text):
    """Remove Hebrew punctuation from the given text."""
    hebrew_punctuation = r"[\u0591-\u05C7]+"
    heb_no_nikud = re.sub(hebrew_punctuation, "", text)
    heb_no_nikud = heb_no_nikud.replace("\uFB2A", "\u05E9").replace(
        "\uFB2B", "\u05E9"
    )  # שׁ to ש
    return heb_no_nikud


def get_raw_text_by_sentence(samples, sample_names, lex=True) -> pd.DataFrame:
    book = [s.split(":")[0] for s in sample_names]

    df = pd.DataFrame(zip(book, sample_names), columns=["book", "sentence_path"])
    text_samples_lex = samples_to_text(samples, lex=lex)
    text_samples = samples_to_text(samples, lex=False)
    df["text_lex"] = text_samples_lex
    df["text"] = text_samples
    df["n_words_lex"] = [len(s.split(" ")) for s in text_samples_lex]
    df["n_words"] = [len(s.split(" ")) for s in text_samples]
    return df


def get_majority_bib(samples):
    # some scrolls are not determinsitc (really low number)
    all_dicts = [d for sublist in samples for d in sublist]
    bib_counter = Counter(d["bib"] for d in all_dicts if "bib" in d)
    majority_bib = bib_counter.most_common(1)[0][0]
    return majority_bib


def process_scrolls_to_features(
    chunk_size, filtered_data, pre_processing_tasks, max_overlap=10
) -> pd.DataFrame:
    features_by_sample_dfs = []
    wc = count_words_from_corpus(filtered_data, pre_processing_tasks)
    logger.info(f"Total words in corpus: {sum(wc.values()):,}")
    for scroll_name, book_data in tqdm(filtered_data.items()):
        if pre_processing_tasks:
            book_data = pre_processing(book_data, pre_processing_tasks, wc)
        samples, sample_names = parser_data.chunk_by_scroll(
            book_data, word_per_samples=chunk_size, max_overlap=max_overlap
        )
        if not samples or not sample_names:
            logger.info(f"{scroll_name} is empty")
            continue

        raw_txt = get_raw_text_by_sentence(samples, sample_names)
        starr_features = starr.get_starr_features(samples)
        tmp_df = pd.concat([raw_txt, starr_features], axis=1)
        tmp_df["bib"] = get_majority_bib(samples)
        features_by_sample_dfs.append(tmp_df)

    return pd.concat(features_by_sample_dfs, ignore_index=True)


def filter_df_by_rules(df_origin):
    df = df_origin
    df["n_words"] = df["text"].str.split().apply(len)
    # Filter out non-Hebrew books
    df = df[~df["book"].isin(NOT_HEB_BOOKS)]

    # Filter out non-bib books
    df = df[df["bib"] == "nonbib"]

    # Aggregate text by book
    df_grouped = df.groupby("book")["text"].apply(" ".join).reset_index()
    df_grouped["number_of_words"] = df_grouped["text"].str.split().apply(len)

    # Filter books with at least MIN_WORDS_PER_SCROLL words
    df_by_book = df_grouped[df_grouped["number_of_words"] >= MIN_WORDS_PER_SCROLL]
    books_with_enough_words = df_by_book["book"].tolist()
    df_final = df[df["book"].isin(books_with_enough_words)]

    # Remove scrolls that were manually removed
    df_final = df_final[~df_final["book"].isin(manually_remove_scrolls)]

    # Filter out chunks with less than 10 words
    df_final = df_final[df_final["n_words"] > 15]
    return df_final


def add_sectarian_label(df):
    import yaml

    with open(f"{BASE_DIR}/data/yamls/all_sectarian_texts.yaml", "r") as f:
        all_sectarian_texts = yaml.load(f, Loader=yaml.FullLoader)
        all_sectarian_texts = {
            k: v for k, v in all_sectarian_texts.items() if len(v) > 0
        }

    flatten = []
    for section in all_sectarian_texts.keys():
        for scroll in all_sectarian_texts[section].keys():
            for book in all_sectarian_texts[section][scroll]:
                flatten.append({"section": section, "scroll": scroll, "book": book})

    books_with_label = pd.DataFrame(flatten)

    df_with_label = pd.merge(df, books_with_label, how="outer", on="book")
    df_with_label = df_with_label.drop("scroll", axis=1)
    return df_with_label


def generate_composition_to_book():
    with open(
        "/Users/yonatanlou/dev/QumranNLP/data/yamls/all_texts_by_composition.yaml"
    ) as f:
        all_texts_by_composition = yaml.safe_load(f)

    df_list = []
    for key, value in all_texts_by_composition.items():
        temp_df = pd.DataFrame(value, columns=["book"])
        temp_df["composition"] = key
        df_list.append(temp_df)

    # Concatenate all DataFrames
    df = pd.concat(df_list, ignore_index=True)
    return df


def add_labels(df):
    labels = pd.read_csv(f"{BASE_DIR}/data/qumran_labels.csv")
    df = pd.merge(
        df, labels[["book", "section", "composition", "genre"]], on="book", how="left"
    )
    return df


def not_conj_or_prepoistion(d):
    sp = d["parsed_morph"]["sp"]
    cl = d["parsed_morph"]["cl"]
    if sp == "ptcl" and cl == "conj":
        return False
    else:
        return True


def lemmatization(sample):
    if not_conj_or_prepoistion(sample):
        return sample
    else:
        return None


from collections import Counter


def count_words_from_corpus(filtered_data, pre_processing_tasks):
    word_counter = Counter()

    for book, word_list in filtered_data.items():
        for word_dict in word_list:
            if not_conj_or_prepoistion(word_dict):
                word = word_dict.get("transcript", "")
                if word:
                    word = word.strip()
                    word_counter[word] += 1

    return word_counter


def apply_task(sample, task, wc=None):
    if task == "LEMMATIZATION":
        return lemmatization(sample)
    elif "MFW" in task:
        mfw_thresh = int(task.split("=")[1])
        return remove_mfw(sample, wc, n=mfw_thresh)
    elif task == "STOPWORDS":
        return remove_stop_words(sample)
    elif task == "LEX":
        return sample  # handling it after
    else:
        raise ValueError(
            "pre_processing_tasks must contain one or more of: MFW, LEMMATIZATION, STOPWORDS"
        )


def pre_processing(book_data, pre_processing_tasks, wc):
    new_book_data = []
    for sample in book_data:
        for task in pre_processing_tasks:
            sample = apply_task(sample, task, wc)
            if sample is None:
                break
        if sample is not None:
            new_book_data.append(sample)
    return new_book_data


def pre_processing_post_chunking(samples, pre_processing_tasks, wc):
    default_sample = samples[0][0]
    new_samples = []
    for chunk in samples:
        new_chunk = []
        for sample in chunk:
            for task in pre_processing_tasks:
                sample = apply_task(sample, task, wc)
                if sample is None:
                    break
            if sample is not None:
                new_chunk.append(sample)
        if len(new_chunk) > 0:
            new_samples.append(new_chunk)
        else:
            new_samples.append([default_sample])
    return new_samples


def remove_mfw(sample, wc, n=100):
    wc_top_n = [w[0] for w in wc.most_common(n)]

    word = sample.get("transcript", "")
    if word:
        word = word.strip()
        if word in wc_top_n:
            return None
        else:
            return sample


def remove_stop_words(sample):
    # using only nouns, adjectives, verbs, and adverbs.
    sp = sample["parsed_morph"]["sp"]
    cl = sample["parsed_morph"]["cl"]
    if (sp in ["subs", "adjv", "verb"]) or (cl == "advb"):
        return sample
    else:
        return None
