import pandas as pd
import re
import yaml
MIN_WORDS_PER_BOOK = 100
MIN_WORD_PER_LINE = 5


chars_to_delete = re.compile("[\\\\\^><»≥≤/?Ø\\]\\[«|}{]")
def get_biblical_from_line(line):
    """
    Returns the biblical section of a line.
    """
    bib = F.biblical.v(line)
    if bib == None:
        return 'nonbib'
    elif bib == 1:
        return 'bib'
    elif bib == 2:
        return 'biblical_non_biblical'

def remove_chars(s):
    chars_to_delete = "#ε^><»≥≤/?Ø«|}{׳"
    for char in chars_to_delete:
        s = s.replace(char, "")
    return s


def parse_data(data):
    lst_for_df = []
    for book in data.keys():
        for line in data[book]:
            line_splitted = line[0].split("\t")
            book = line_splitted[0].split(" ")[0]
            pasuk = line_splitted[0].split(" ")[1]
            bib_nonbib = line_splitted[1]
            text = line_splitted[2]
            lst_for_df.append(
                {"book": book, "pasuk": pasuk, "label": bib_nonbib, "text": text}
            )
    df = pd.DataFrame(lst_for_df)
    return df


def write_data(data, filename):
    with open(filename, "w") as f:
        for book in data.keys():
            for line in data[book]:
                line = "".join(line[0]) + "\n"

                f.write(line)


def generate_stats(df):
    df["word_count"] = df["text"].str.split(" ").str.len()
    book_stats = (
        df.groupby("book")["word_count"].agg(["sum", "count"]).sort_values(by="count")
    )
    book_stats.columns = ["word_count", "lines_count"]
    label_stats = (
        df.groupby("label")["word_count"].agg(["sum", "count"]).sort_values(by="count")
    )
    label_stats.columns = ["word_count", "lines_count"]
    return book_stats, label_stats


import matplotlib.pyplot as plt


def plot_word_line_stats(book_stats, label_stats):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    book_stats["word_count"].hist(bins=100, ax=ax[0, 0])
    book_stats["lines_count"].hist(bins=100, ax=ax[0, 1])
    label_stats["word_count"].plot(kind="bar", ax=ax[1, 0])
    label_stats["lines_count"].plot(kind="bar", ax=ax[1, 1])
    ax[0, 0].set_title("Word Count per book")
    ax[0, 1].set_title("Lines Count per book")
    ax[1, 0].set_title("Word Count per label")
    ax[1, 1].set_title("Lines Count per label")


def data_cleaning(df, book_stats):
    books_with_enough_data = (
        book_stats[book_stats["word_count"] >= MIN_WORDS_PER_BOOK]
        .sort_values(by="word_count", ascending=False)
        .index.to_list()
    )
    df_filtered = df[df["book"].isin(books_with_enough_data)]
    # df_filtered = df_filtered[df_filtered["label"] != "biblical_non_biblical"]
    # df_filtered = df_filtered[df_filtered["word_count"] >= MIN_WORD_PER_LINE]
    df_filtered["book_pasuk"] = df_filtered["book"] + " " + df_filtered["pasuk"]
    return df_filtered

def generate_composition_to_book():
    with open("/Users/yonatanlou/dev/QumranNLP/data/yamls/all_texts_by_composition.yaml") as f:
        all_texts_by_composition = yaml.safe_load(f)


    df_list = []
    for key, value in all_texts_by_composition.items():
        temp_df = pd.DataFrame(value, columns=['book'])
        temp_df['composition'] = key
        df_list.append(temp_df)

    # Concatenate all DataFrames
    df = pd.concat(df_list, ignore_index=True)
    return df
