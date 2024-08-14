import re
from collections import Counter

import pandas as pd
import yaml
from tqdm import tqdm

from config import BASE_DIR
from notebooks.constants import NOT_HEB_BOOKS
from src.features.Starr import starr
from src.parsers import parser_data

ALLOWED_CHARS = "אבגדהוזחטיכלמנסעפצקרשתםןףךץ. 1234567890"
MIN_WORDS_PER_SCROLL = 300


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
    filtered_data, chunk_size, max_overlap=10
) -> pd.DataFrame:
    features_by_sample_dfs = []

    for scroll_name, book_data in tqdm(filtered_data.items()):
        samples, sample_names = parser_data.chunk_by_scroll(
            book_data, word_per_samples=chunk_size, max_overlap=max_overlap
        )

        if not samples or not sample_names:
            print(f"empty: {scroll_name}")
            continue
        raw_txt = get_raw_text_by_sentence(samples, sample_names)
        starr_features = starr.get_starr_features(samples)
        tmp_df = pd.concat([raw_txt, starr_features], axis=1)
        tmp_df["bib"] = get_majority_bib(samples)
        features_by_sample_dfs.append(tmp_df)

    return pd.concat(features_by_sample_dfs, ignore_index=True)


manually_remove_scrolls = [
    "4Q" + str(i)
    for i in list(range(196, 200))
    + list(range(201, 215))
    + list(range(242, 246))
    + [318]
    + list(range(529, 570))
    + list(range(342, 360)) +
    ["4Q249Z"]
]


def filter_df_by_rules(df_origin):
    df = add_sectarian_label(df_origin)

    # Merge with the composition-to-book mapping and remove duplicates
    composition_to_book = generate_composition_to_book().drop_duplicates(
        subset=["book"]
    )
    df = pd.merge(
        df, composition_to_book, on="book", how="left", validate="many_to_one"
    )

    # Filter out non-Hebrew books
    df = df[~df["book"].isin(NOT_HEB_BOOKS)]

    df = df[df["bib"] == "nonbib"]
    # Aggregate text by book
    df_grouped = df.groupby("book")["text"].apply(" ".join).reset_index()

    # Merge with additional book info and remove duplicates
    book_info = df[["book", "composition", "section"]].drop_duplicates()
    df_by_book = pd.merge(df_grouped, book_info, on="book", how="inner")

    # Compute the number of words in each book
    df_by_book["number_of_words"] = df_by_book["text"].str.split().apply(len)

    # Filter books with at least 300 words
    df_by_book = df_by_book[df_by_book["number_of_words"] >= MIN_WORDS_PER_SCROLL]

    # Get the list of books with enough words
    books_with_enough_words = df_by_book["book"].tolist()

    # Filter the original dataframe to keep only books with enough words
    df_final = df[df["book"].isin(books_with_enough_words)]
    df_final = df_final[~df_final["book"].isin(manually_remove_scrolls)]
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
