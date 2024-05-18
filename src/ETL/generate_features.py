from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from src.features.BERT.bert import aleph_bert_preprocessing
from config import BASE_DIR
from src.features.Starr import starr
from src.hierarchial_clustering.constants import WORD_PER_SAMPLES

from src.parsers import parser_data
from src.parsers.MorphParser import MorphParser
from src.parsers.text_reader import read_text


text_file = f"{BASE_DIR}/data/texts/abegg/dss_{bib_nonbib}.txt"
yaml_dir = f"{BASE_DIR}/data/yamls"


def generate_starr_features(bib_nonbib, word_per_sample, text_file_path, yaml_dir):
    morph_parser = MorphParser(yaml_dir=yaml_dir)
    data, lines = read_text(text_file_path)
    filtered_data = defaultdict(list)
    for entry in data:
        if bib_nonbib == "nonbib":
            filtered_data[entry["scroll_name"]].append(entry)
            entry["parsed_morph"] = morph_parser.parse_morph(entry["morph"])
        else:
            pass

    features_by_sample_dfs_lst = []
    for scroll_name, book_data in tqdm(filtered_data.items()):
        samples, sample_names = parser_data.get_samples(
            book_data, word_per_samples=word_per_sample
        )
        if len(samples[-1]) < WORD_PER_SAMPLES:
            samples = samples[:-1]
            sample_names = sample_names[:-1]

        starr_features = starr.get_starr_features_v2(samples)
        heb_trans_samples = aleph_bert_preprocessing(samples)
        starr_features["book"], starr_features["sentence_path"] = [
            s.split(":")[0] for s in sample_names
        ], sample_names
        starr_features["processed_text"] = heb_trans_samples
        starr_features["n_words"] = [len(s.split(" ")) for s in heb_trans_samples]
        features_by_sample_dfs_lst.append(starr_features)
        df = pd.concat(features_by_sample_dfs_lst)
        return df


if __name__ == "__main__":
    df = generate_starr_features(
        bib_nonbib="nonbib", word_per_sample=100, text_file=text_file, yaml_dir=yaml_dir
    )
    df.to_csv(
        f"{BASE_DIR}/data/starr_features_nonbib_dss_100_words_window.csv", index=False
    )
