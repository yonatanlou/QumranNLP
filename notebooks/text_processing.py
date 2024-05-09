import os

from config import BASE_DIR

import re
import pandas as pd
import matplotlib.pyplot as plt
from notebooks.utils import parse_data, generate_stats
from notebooks.constants import MIN_WORDS_PER_BOOK
from notebooks.utils import data_cleaning

ALLOWED_CHARS = "אבגדהוזחטיכלמנסעפצקרשתםןףךץ. 1234567890"
chars_to_delete = re.compile("[\\\\\^><»≥≤/?Ø\\]\\[«|}{]")
from logger import get_logger

logger = get_logger(__name__)

import tqdm


def get_biblical_from_line(line):
    """
    Returns the biblical section of a line.
    """
    bib = F.biblical.v(line)
    if bib == None:
        return "nonbib"
    elif bib == 1:
        return "bib"
    elif bib == 2:
        return "biblical_non_biblical"


def remove_chars(s):
    chars_to_delete = "#ε^><»≥≤/?Ø«|}{׳"
    for char in chars_to_delete:
        s = s.replace(char, "")
    return s


def generate_raw_data():
    from tf.app import use

    A = use("ETCBC/dss", hoist=globals())
    data = {}
    for line in tqdm.tqdm(F.otype.s("line")[:]):
        book_and_chapter = A.sectionStrFromNode(line)
        book = A.sectionStrFromNode(line).split(" ")[0]
        text = (
            book_and_chapter
            + "\t"
            + str(get_biblical_from_line(line))
            + "\t"
            + T.text(line)
        )
        text = remove_chars(text).replace("\xa0", "").replace("׃", ".")
        text = [text]
        if book not in data:
            data[book] = [text]
        else:
            data[book].append(text)
    return data
### Future use: how to get the part-of-speech data
# data = []
# for w in tqdm(F.otype.s("word")[:]):
#     book_and_chapter = A.sectionStrFromNode(w)
#     book = A.sectionStrFromNode(w).split(" ")[0]
#     text = ( T.text(w)
#     )
#     text = remove_chars(text).replace("\xa0", "").replace("׃", ".")
#     sp = F.sp.v(w)
#     res = {"book":book, "text": text, "sp": sp}
#
#     data.append(res)


def generate_corpus_df() -> pd.DataFrame:
    data = generate_raw_data()
    df = parse_data(data)
    book_stats, label_stats = generate_stats(df)
    logger.info(f"Generated {df['book'].nunique()} unique books")
    df_filtered = data_cleaning(df, book_stats)
    logger.info(
        f"Removed {df['book'].nunique()-df_filtered['book'].nunique()} books that are smaller than {MIN_WORDS_PER_BOOK} words per book"
    )
    return df_filtered


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
    return df_with_label


def convert_df_to_by_book(df):
    by_book = df.groupby("book")["text"].apply(list).str.join(" ")
    df_by_book = pd.merge(
        by_book, df[["book", "label", "section"]], on="book", how="inner"
    ).drop_duplicates()
    return df_by_book


def doc_level_pre_process(doc):
    doc = doc.replace("╱", "")
    doc = re.sub(r"\s+", " ", doc)
    return doc


def replace_for_ot_sofit(word):
    OT_SOFIT = {"מ": "ם", "נ": "ן", "פ": "ף", "צ": "ץ", "כ": "ך"}
    last_char = word[-1]
    if last_char in OT_SOFIT.keys():
        word = word[:-1] + OT_SOFIT[last_char]
    return word


def remove_not_heb_chars(word):
    new_word = []
    removed_chars = set()
    for char in word:
        if char in ALLOWED_CHARS:
            new_word.append(char)
        if char not in ALLOWED_CHARS:
            removed_chars.add(char)
    return "".join(new_word), removed_chars





def pre_process_corpus(df_by_book: pd.DataFrame,stop_words, remove_stop_words=True):
    all_docs = []
    tqdm_flex = tqdm.tqdm
    if os.environ["JPY_SESSION_NAME"].endswith("ipynb"):
        tqdm_flex = tqdm.tqdm_notebook
    for i, doc in tqdm_flex(enumerate(df_by_book["text"])):
        doc = doc_level_pre_process(doc)
        tmp_words_list = []
        word_replace_counter = 0
        chars_removed = set()
        for word in doc.split():
            if remove_stop_words:
                if word in stop_words:
                    continue
            word = replace_for_ot_sofit(word)
            new_word, char_removed = remove_not_heb_chars(word)
            chars_removed.update(char_removed)
            if word != new_word:
                word_replace_counter += 1
            tmp_words_list.append(new_word)
        if word_replace_counter:
            print(
                f"({i}) replaced {word_replace_counter} words in {df_by_book.iloc[i, :].book} ({word_replace_counter / len(doc.split()):.3f} from all words). chars removed: {chars_removed}")
        doc = " ".join(tmp_words_list)
        all_docs.append(doc)
    return all_docs

if __name__ == "__main__":
    pass
