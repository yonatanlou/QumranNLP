import yaml
from os import path
from collections import defaultdict
from datetime import datetime
from logger import get_logger

logger = get_logger(__name__)

root_path = path.dirname(__file__)


def read_yaml(fileName):
    with open(fileName) as y:
        y = yaml.load(y, Loader=yaml.FullLoader)
    return y


def filter_data_by_field(field, books_list, unfiltered_data):
    c_to_book = {v: k for k, l in books_list.items() for v in l}
    # c_to_book = {b: b for b in books_list}
    filtered_data = defaultdict(list)
    for entry in unfiltered_data:
        if entry[field] in c_to_book.keys():
            filtered_data[c_to_book[entry[field]]].append(entry)
    return filtered_data


class Transcriptor:
    def __init__(self, transcription_yaml_path):
        with open(transcription_yaml_path, "r") as f:
            self.trans_dict = yaml.safe_load(f)

    def latin_to_heb(self, latin_text, entry=None):
        s = ""
        for x in latin_text:
            if x in self.trans_dict["latin_to_heb"].keys():
                s += self.trans_dict["latin_to_heb"][x]
            else:
                logger.info(
                    f"couldnt translate {x} to Hebrew (full word is {latin_text}), current s:{s},  ({entry['book_name']}, {entry['chapter_name']})"
                )
        return s


def get_time():
    return datetime.today().strftime("%d_%m_%Y__%H_%M_%S")
