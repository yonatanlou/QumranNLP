import re

import pandas as pd
from tqdm import tqdm

from src.features.Starr import starr
from src.parsers import parser_data

ALLOWED_CHARS = "אבגדהוזחטיכלמנסעפצקרשתםןףךץ. 1234567890"


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
    heb_no_nikud = heb_no_nikud.replace("\uFB2A", "\u05E9")  # שׁ to ש
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


def process_scrolls_to_features(
    filtered_data, word_per_samples, sentence_divider
) -> pd.DataFrame:
    features_by_sample_dfs = []

    for scroll_name, book_data in tqdm(filtered_data.items()):
        samples, sample_names = parser_data.get_samples(
            book_data,
            word_per_samples=word_per_samples,
            sentence_divider=sentence_divider,
        )

        if not samples or not sample_names:
            continue
        raw_txt = get_raw_text_by_sentence(samples, sample_names)
        starr_features = starr.get_starr_features(samples)
        tmp_df = pd.concat([raw_txt, starr_features], axis=1)
        features_by_sample_dfs.append(tmp_df)

    return pd.concat(features_by_sample_dfs, ignore_index=True)
