from collections import Counter

import numpy as np
from src.parsers.MorphParser import MorphParser
from config import BASE_DIR
from src.parsers.text_reader import read_text
from utils import filter_data_by_field, read_yaml
from logger import get_logger

logger = get_logger(__name__)


def split_data_to_samples(entries, n_words_per_feature):
    words = []
    word = []
    word_num = None
    for entry in entries:
        if entry["word_line_num"] != word_num or "sp" not in entry["parsed_morph"]:
            if entry["transcript"] == ".":
                word.append(entry)
                words.append(word)
                word = []
                continue
            if len(word) != 0:
                words.append(word)
            word = []
            word_num = entry["word_line_num"]
            if "sp" in entry["parsed_morph"]:
                word.append(entry)
            continue
        word.append(entry)
    samples = [
        words[i : i + n_words_per_feature]
        for i in range(0, len(words), n_words_per_feature)
    ]
    return [[w for word in sample for w in word] for sample in samples]


def gen_samples(entries, n_words_per_feature, sentence_divider):
    def count_words(entries):
        word_count = 0
        for entry in entries:
            if entry["sub_word_num"] == "1" and "sp" in entry["parsed_morph"]:
                word_count += 1
        return word_count

    def gen_sample(entries, sentence_divider):
        curr_word_entries = []
        for e, entry in enumerate(entries):
            if "sp" not in entry.get("parsed_morph", ""):
                curr_word_entries = []
                continue
            else:
                if e != len(entries) - 1 and entries[e + 1].get(
                    "word_line_num"
                ) == entry.get("word_line_num"):
                    curr_word_entries.append(entry)
                    if (
                        e + 1 < len(entries)
                        and entries[e + 1].get("transcript") == sentence_divider
                    ):
                        curr_word_entries.append(entries[e + 1])
                else:
                    curr_word_entries.append(entry)
                    if (
                        e + 1 < len(entries)
                        and entries[e + 1].get("transcript") == sentence_divider
                    ):
                        curr_word_entries.append(entries[e + 1])
                    yield curr_word_entries
                    curr_word_entries = []

    num_words = count_words(entries)
    if n_words_per_feature:
        num_samples = np.floor(num_words / n_words_per_feature)
    else:
        num_samples = num_words - 1
    sample_generator = gen_sample(entries, sentence_divider)
    finish = False
    i = 0
    while not finish:
        curr_sample = []
        for j in range(n_words_per_feature):
            try:
                curr_sample.extend(next(sample_generator))
            except StopIteration:
                # print("first", i, j, num_samples)
                finish = True
                break
        if len(curr_sample) == 0:
            # print("second", i, j, num_samples, len(curr_sample))
            sample_name = ""
        else:
            sample_name = f"{curr_sample[0]['scroll_name']}:{curr_sample[0]['frag_label']}:{curr_sample[0]['frag_line_num']}-{curr_sample[-1]['frag_label']}:{curr_sample[-1]['frag_line_num']}"
        i += 1
        yield curr_sample, sample_name

    # for i in np.arange(num_samples):
    #     curr_sample = []
    #     for j in range(n_words_per_feature):
    #         print(j)
    #         curr_sample.extend(next(sample_generator))
    #         # try:
    #         #     curr_sample.extend(next(sample_generator))
    #         # except StopIteration:
    #         #     a=1
    #         #     print("first", i, j, num_samples)
    #     if len(curr_sample) == 0:
    #         a=1
    #         print("second", i, j, num_samples), len(curr_sample)
    #     else:
    #         sample_name = f"{curr_sample[0]['scroll_name']}:{curr_sample[0]['frag_label']}:{curr_sample[0]['frag_line_num']}-{curr_sample[-1]['frag_label']}:{curr_sample[-1]['frag_line_num']}"
    #     yield curr_sample, sample_name


def get_samples(book_data, word_per_samples=25, sentence_divider="."):
    if len(book_data) == 0:
        print("empty")
        return None, None

    res = zip(*[x for x in gen_samples(book_data, word_per_samples, sentence_divider)])
    try:
        samples, sample_names = res
    except Exception as e:
        print(f"wrong: {e}")
        return None, None
    # samples1 = split_data_to_samples(book_data, WORDS_PER_SAMPLE)
    # [i for i in range(len(samples)) if len(samples[i]) != len(samples1[i])]
    return samples, sample_names


def get_dss_data(books_list, type="nonbib"):
    text_file = f"{BASE_DIR}/data/texts/abegg/dss_{type}.txt"
    yaml_dir = f"{BASE_DIR}/data/yamls"
    morph_parser = MorphParser(yaml_dir=yaml_dir)

    data, lines = read_text(text_file)
    num_of_lines_per_book = "".join(
        [
            f"{book[0]}: {book[1]},"
            for book in Counter(line[0] for line in lines).most_common()
        ]
    )
    logger.info(f"book_sizes: {num_of_lines_per_book}")
    filter_field = "scroll_name" if type == "nonbib" else "book_name"
    if books_list:
        filtered_data = filter_data_by_field(filter_field, books_list, data)
    else:
        filtered_data = data
    logger.info("processed the following books}")
    for book in books_list:
        print(book, end=",")
        for entry in filtered_data[book]:
            entry["parsed_morph"] = morph_parser.parse_morph(entry["morph"])
    return filtered_data


def get_all_data():
    text_file = f"{BASE_DIR}/data/texts/abegg/dss_nonbib.txt"
    yaml_dir = f"{BASE_DIR}/data/yamls"
    return read_text(text_file, yaml_dir)
