import re
import logger
import string

import pandas as pd
import yaml
from tqdm.auto import tqdm

from config import BASE_DIR
from notebooks.constants import NOT_HEB_BOOKS
from src.features.Starr import starr
from src.parsers import parser_data
from logger import get_logger

logger = get_logger(__name__)
ALLOWED_CHARS = "אבגדהוזחטיכלמנסעפצקרשתםןףךץ. 1234567890"
MIN_WORDS_PER_SCROLL = 300
MFW_THRESHOLD = 10
manually_remove_scrolls = [
    # aramic:
    "1Q20",
    "1Q21",
    "1Q23",
    "1Q24",
    "1Q32",
    "2Q24",
    "2Q26",
    "4Q318",
    "4Q339",
    "4Q523",
    "11Q18",
    "4Q201",
    "4Q202",
    "4Q203",
    "4Q204",
    "4Q205",
    "4Q206",
    "4Q207",
    "4Q208",
    "4Q209",
    "4Q21",
    "4Q210",
    "4Q211",
    "4Q212",
    "4Q213",
    "4Q213a",
    "4Q213b",
    "4Q214",
    "4Q214a",
    "4Q214b",  # 4Q201 to 4Q214b
    *["4Q" + str(i) for i in range(196, 200)],  # 4Q196 to 4Q199
    *["4Q" + str(i) for i in range(242, 246)],  # 4Q242 to 4Q245
    *["4Q" + str(i) for i in range(529, 570)],  # 4Q529 to 4Q569
    # too short and fragmentary
    *["1Q" + str(i) for i in range(41, 71)],  # 1Q41 to 1Q70
    *["2Q" + str(i) for i in range(22, 33)],  # 2Q22 to 2Q32
    *["4Q" + str(i) for i in range(234, 236)],  # 4Q234, 4Q235
    "4Q238",
    "4Q249",
    "4Q250",
    "4Q313a",
    "4Q313b",
    *[
        "4Q" + str(i) + suffix
        for i in range(281, 283)
        for suffix in [""] + list(string.ascii_lowercase)
    ],  # 4Q281 to 4Q282 with all subsections
    *[
        "4Q" + str(i) + suffix
        for i in range(291, 295)
        for suffix in [""] + list(string.ascii_lowercase)
    ],
    *[
        "4Q" + str(i) + suffix
        for i in range(249, 300)
        for suffix in [""] + list(string.ascii_lowercase)
    ],  # 4Q249 with all subsections
    "4Q307",
    "4Q313",
    "4Q338",
    "4Q340",
    *[
        "4Q" + str(i) + suffix for i in range(360, 361) for suffix in ["", "a"]
    ],  # 4Q360, 4Q360a
    *["4Q" + str(i) for i in range(441, 448)],  # 4Q441 to 4Q447
    *["4Q" + str(i) for i in range(449, 460)],  # 4Q449 to 4Q459
    "4Q464a",
    "4Q464b",
    "4Q465",
    *[
        "4Q" + str(i) + suffix
        for i in range(466, 469)
        for suffix in [""] + list(string.ascii_lowercase)
    ],  # 4Q466 to 4Q468 with all subsections
    "4Q468aa",
    "4Q468bb",
    "4Q468cc",
    "4Q468dd",
    "4Q469",
    "4Q471a",
    "4Q471n",
    "4Q272",
    "4Q473",
    *[
        "4Q" + str(i) + suffix
        for i in range(478, 482)
        for suffix in [""] + list(string.ascii_lowercase)
    ],  # 4Q478 to 4Q481 with all subsections
    *["4Q" + str(i) for i in range(484, 490)],  # 4Q484 to 4Q489
    *["4Q" + str(i) for i in range(498, 501)],  # 4Q498 to 4Q500
    *["4Q" + str(i) for i in range(515, 521)],  # 4Q515 to 4Q520
    *["4Q" + str(i) for i in range(526, 529)],  # 4Q526 to 4Q528
    *["4Q" + str(i) for i in range(570, 588)],  # 4Q570 to 4Q587
    "11Q15",
    "11Q16",
    *["11Q" + str(i) for i in range(22, 28)],  # 11Q22 to 11Q27
    *["3Q" + str(i) for i in range(1, 15) if i != 15],  # Delete all 3Q except 3Q15
    *[
        "4Q" + str(i) for i in range(341, 360)
    ],  # 4Q341 to 4Q359 (probably not from Qumran)
    *[
        "5Q" + str(i) for i in range(1, 26) if i not in [12, 13]
    ],  # Delete all 5Q except 5Q12, 5Q13
    *["6Q" + str(i) for i in range(1, 31) if i != 15],  # Delete all 6Q except 6Q15
    *["8Q" + str(i) for i in range(1, 6)],  # Delete all 8Q
    *["9Q" + str(i) for i in range(1, 10)],  # Delete all 9Q
    *["10Q" + str(i) for i in range(1, 10)],  # Delete all 10Q
    "Pam43113",
    "Pam43124",
    "PAM43660",
    "PAM43661",
    "PAM43663",
    "PAM43664",
    "PAM43665",
    "PAM43666",
    "PAM43667",
    "PAM43668",
    "PAM43669",
    "PAM43670",
    "PAM43671",
    "PAM43672",
    "PAM43673",
    "PAM43674",
    "PAM43675",
    "PAM43676",
    "PAM43677",
    "PAM43678",
    "PAM43679",
    "PAM43680",
    "PAM43682",
    "PAM43683",
    "PAM43684",
    "PAM43685",
    "PAM43686",
    "PAM43688",
    "PAM43689",
    "PAM43690",
    "PAM43691",
    "PAM43692",
    "PAM43693",
    "PAM43694",
    "PAM43695",
    "PAM43696",
    "PAM43697",
    "PAM43698",
    "PAM43699",
    "PAM43700",
    "PAM43701",
    "PAM44102",  # Delete all PAM entries
    "Xq1",
    "Xq2",
    "Xq3",
    "XQ6",
    "XQ7",
    "XQ8",  # Delete XQ, KhQ
]


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
    # Filter out non-Hebrew books
    df = df_origin
    df = df[~df["book"].isin(NOT_HEB_BOOKS)]

    df = df[df["bib"] == "nonbib"]
    # Aggregate text by book
    df_grouped = df.groupby("book")["text"].apply(" ".join).reset_index()
    df_grouped["number_of_words"] = df_grouped["text"].str.split().apply(len)

    # Filter books with at least MIN_WORDS_PER_SCROLL words
    df_by_book = df_grouped[df_grouped["number_of_words"] >= MIN_WORDS_PER_SCROLL]
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
