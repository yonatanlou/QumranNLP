from config import BASE_DIR

import re
import pandas as pd
import matplotlib.pyplot as plt
from notebooks.utils import parse_data, generate_stats, MIN_WORDS_PER_BOOK
from notebooks.utils import data_cleaning
chars_to_delete = re.compile("[\\\\\^><»≥≤/?Ø\\]\\[«|}{]")
from logger import get_logger
logger = get_logger(__name__)

from tqdm import tqdm


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
    for line in tqdm(F.otype.s("line")[:]):
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

def generate_corpus_df() -> pd.DataFrame:
    data = generate_raw_data()
    df = parse_data(data)
    book_stats, label_stats = generate_stats(df)
    logger.info(f"Generated {df['book'].nunique()} unique books")
    df_filtered = data_cleaning(df, book_stats)
    logger.info(f"Removed {df['book'].nunique()-df_filtered['book'].nunique()} books that are smaller than {MIN_WORDS_PER_BOOK} words per book")
    return df_filtered

def add_sectarian_label(df):
    import yaml

    with open(f"{BASE_DIR}/data/yamls/all_sectarian_texts.yaml", "r") as f:
        all_sectarian_texts = yaml.load(f, Loader=yaml.FullLoader)
        all_sectarian_texts = {k: v for k, v in all_sectarian_texts.items() if len(v) > 0}

    flatten = []
    for section in all_sectarian_texts.keys():
        for scroll in all_sectarian_texts[section].keys():
            for book in all_sectarian_texts[section][scroll]:
                flatten.append({"section": section, "scroll": scroll, "book": book})

    books_with_label = pd.DataFrame(flatten)

    df_with_label = pd.merge(df,books_with_label, how="outer", on="book")
    return df_with_label

def convert_df_to_by_book(df):
    by_book = df.groupby("book")["text"].apply(list).str.join(" ")
    df_by_book = pd.merge(by_book, df[["book","label","section"]], on="book", how="inner").drop_duplicates()
    return df_by_book





if __name__ == '__main__':
    pass
