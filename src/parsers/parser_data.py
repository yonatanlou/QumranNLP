from collections import Counter

import numpy as np
from src.parsers.MorphParser import MorphParser
from config import BASE_DIR
from src.parsers.text_reader import read_text
from base_utils import filter_data_by_field, read_yaml
from logger import get_logger

logger = get_logger(__name__)
MISSING_SIGN = "ε"
UNCERTAIN_SIGN_1 = "?"
UNCERTAIN_SIGN_2 = "#"
UNCERTAIN_SIGNS = [MISSING_SIGN, UNCERTAIN_SIGN_1, UNCERTAIN_SIGN_2]


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


def counting_as_word_statement(e):
    if (e["parsed_morph"]["cl"] == "conj") or (e["parsed_morph"]["cl"] == "prep"):
        return False
    if e["transcript"] == ".":
        return False
    if e["transcript"].strip() in UNCERTAIN_SIGNS:
        return False
    else:
        return True


def chunking(entries, chunk_size, max_overlap=10):
    samples = []
    sample_names = []
    current_sample = []
    word_count = 0
    current_frag_line_num = None
    overlap_entries = []
    # uncertain_words_count = 0
    # tmp_frag_size = 0

    for i, entry in enumerate(entries):
        # Skip entries without 'part-of-speec' in parsed_morph
        if "sp" not in entry.get("parsed_morph", ""):
            continue

        # Check if we're starting a new fragment line
        if entry["frag_line_num"] != current_frag_line_num:
            # If we have a current sample and we've exceeded chunk_size
            if current_sample and word_count > chunk_size:
                samples.append(current_sample)
                sample_name = f"{current_sample[0]['scroll_name']}:{current_sample[0]['frag_label']}:{current_sample[0]['frag_line_num']}-{current_sample[-1]['frag_label']}:{current_sample[-1]['frag_line_num']}"
                sample_names.append(sample_name)

                # Start a new sample with the overlap
                if len(overlap_entries) > max_overlap:
                    overlap_entries = overlap_entries[-max_overlap:]
                current_sample = overlap_entries + [entry]
                overlap_word_count = len(
                    [i for i in overlap_entries if counting_as_word_statement(i)]
                )
                word_count = overlap_word_count + 1
                overlap_entries = []
            else:
                current_sample.append(entry)
                if counting_as_word_statement(entry):
                    word_count += 1
                # if entry["transcript"].strip() in [MISSING_SIGN, UNCERTAIN_SIGN_1, UNCERTAIN_SIGN_2]:
                #     uncertain_words_count += 1
                # tmp_frag_size +=1

            current_frag_line_num = entry["frag_line_num"]
            overlap_entries = [entry]
        else:
            current_sample.append(entry)
            if counting_as_word_statement(entry):
                word_count += 1
            overlap_entries.append(entry)

    # Add the last sample if it's not empty
    if current_sample:
        samples.append(current_sample)
        sample_name = f"{current_sample[0]['scroll_name']}:{current_sample[0]['frag_label']}:{current_sample[0]['frag_line_num']}-{current_sample[-1]['frag_label']}:{current_sample[-1]['frag_line_num']}"
        sample_names.append(sample_name)

    return samples, sample_names


def chunk_by_scroll(book_data, word_per_samples=25, max_overlap=10):
    if len(book_data) == 0:
        return None, None

    res = zip(*[x for x in chunking(book_data, word_per_samples, max_overlap)])
    try:
        res_not_zipped = [i for i in res]
        samples = [i[0] for i in res_not_zipped]
        sample_names = [i[1] for i in res_not_zipped]
    except Exception as e:
        print(f"wrong: {e}")
        return None, None
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
