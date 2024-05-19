import re

import pandas as pd

from src.features.Starr.features_keys import (
    Feature,
    feature_list,
    methods_name_dict,
    sum_entries,
)
import numpy as np
from src.parsers.MorphParser import MorphParser

alphabet_chars = re.compile("[^a-zA-Z]")
modes = presets = ["average", "ward"]


def get_morph_dict():
    morph_parser = MorphParser()
    pronoun_classes = [
        "noun_bound_pronoun",
        "verb_bound_pronoun",
        "preposition_bound_pronoun",
        "object_marker_bound_pronoun",
        "relative_pronoun",
        "independent_pronoun",
        "interrogative_pronoun",
    ]
    general = ["verbs", "nouns", "words", "particles", "pronouns", "adjectives"]
    specific_word_counts = ["ky", "kya", "hnh", "whnh", "aCr", "oM"]
    morph_name_dict = {
        "noun_classes": {
            x: xx[1]
            for x, xx in morph_parser.raw_morph_yaml["values"]["cl"]["subs"].items()
        },
        "verbal_stems": {
            x: xx[1] for x, xx in morph_parser.raw_morph_yaml["values"]["vs"].items()
        },
        "noun_states": {
            x: xx[1] for x, xx in morph_parser.raw_morph_yaml["values"]["st"].items()
        },
        "verbal_tense": {
            x: xx[1] for x, xx in morph_parser.raw_morph_yaml["values"]["vt"].items()
        },
        "verbal_mood": {
            x: xx[1] for x, xx in morph_parser.raw_morph_yaml["values"]["md"].items()
        },
        "adjective_states": {
            x: xx[1] for x, xx in morph_parser.raw_morph_yaml["values"]["st"].items()
        },
        "pronoun_classes": {x: x for x in pronoun_classes},
        "particle_classes": {
            x: xx[1]
            for x, xx in morph_parser.raw_morph_yaml["values"]["cl"]["ptcl"].items()
        },
        "specific_words": {x: x for x in specific_word_counts},
        "general": {x: x for x in general},
    }
    return morph_name_dict


def check_pronoun(entry):
    parsed_morph = entry["parsed_morph"]

    if parsed_morph["sp"] == "subs" and "sp2" in parsed_morph:
        return "noun_bound_pronoun"
    if parsed_morph["sp"] == "verb" and "sp2" in parsed_morph:
        return "verb_bound_pronoun"
    if parsed_morph["sp"] == "ptcl":
        if parsed_morph["cl"] == "rela":
            return "relative_pronoun"
        if parsed_morph["cl"] == "prep" and "sp2" in parsed_morph:
            return "preposition_bound_pronoun"
        if parsed_morph["cl"] == "objm" and "sp2" in parsed_morph:
            return "object_marker_bound_pronoun"

    if parsed_morph["sp"] == "pron":
        if parsed_morph["cl"] == "indp":
            return "independent_pronoun"
        if parsed_morph["cl"] == "intr":
            return "interrogative_pronoun"


def gen_sample_features(entries, morph_name_dict, feature_list):
    count = {
        key: {k: 0 for k in morph_name_dict[key].values()} for key in morph_name_dict
    }
    word_count_flag = False

    last_filtered_transcript = ""
    for entry in entries:
        filtered_transcript = alphabet_chars.sub("", entry["transcript"])
        if entry["sub_word_num"] == "1":
            word_count_flag = False

        parsed_morph = entry["parsed_morph"]
        if "sp" in parsed_morph:
            if word_count_flag is False:
                count["general"]["words"] += 1
                word_count_flag = True
            if parsed_morph["sp"] == "subs":
                count["general"]["nouns"] += 1
                if (
                    "st" in parsed_morph
                    and entry["parsed_morph"]["st"] in morph_name_dict["noun_states"]
                ):
                    count["noun_states"][
                        morph_name_dict["noun_states"][entry["parsed_morph"]["st"]]
                    ] += 1
                if (
                    entry["parsed_morph"]["cl"]
                    in morph_name_dict["noun_classes"].keys()
                ):
                    count["noun_classes"][
                        morph_name_dict["noun_classes"][entry["parsed_morph"]["cl"]]
                    ] += 1

            if parsed_morph["sp"] == "verb":
                count["general"]["verbs"] += 1
                if (
                    entry["parsed_morph"]["vs"]
                    in morph_name_dict["verbal_stems"].keys()
                ):
                    count["verbal_stems"][
                        morph_name_dict["verbal_stems"][entry["parsed_morph"]["vs"]]
                    ] += 1
                if (
                    "vt" in parsed_morph
                    and entry["parsed_morph"]["vt"] in morph_name_dict["verbal_tense"]
                ):
                    count["verbal_tense"][
                        morph_name_dict["verbal_tense"][entry["parsed_morph"]["vt"]]
                    ] += 1
                    if (
                        "participle"
                        in morph_name_dict["verbal_tense"][entry["parsed_morph"]["vt"]]
                    ):
                        a = 1
                if (
                    "md" in parsed_morph
                    and entry["parsed_morph"]["md"] in morph_name_dict["verbal_mood"]
                ):
                    count["verbal_mood"][
                        morph_name_dict["verbal_mood"][entry["parsed_morph"]["md"]]
                    ] += 1
            if parsed_morph["sp"] == "ptcl":
                count["general"]["particles"] += 1
                count["particle_classes"][
                    morph_name_dict["particle_classes"][entry["parsed_morph"]["cl"]]
                ] += 1

            if parsed_morph["sp"] == "adjv":
                count["general"]["adjectives"] += 1
                if "st" in parsed_morph:
                    count["adjective_states"][
                        morph_name_dict["adjective_states"][entry["parsed_morph"]["st"]]
                    ] += 1

            # count pronouns
            pronoun_type = check_pronoun(entry)
            if pronoun_type:
                count["general"]["pronouns"] += 1
                count["pronoun_classes"][pronoun_type] += 1

            if filtered_transcript in morph_name_dict["specific_words"]:
                if filtered_transcript not in ["om", "oM", "aCr"]:
                    count["specific_words"][filtered_transcript] += 1
                else:
                    if parsed_morph["sp"] == "ptcl":
                        filtered_transcript = (
                            "oM" if filtered_transcript == "om" else filtered_transcript
                        )
                        count["specific_words"][filtered_transcript] += 1

                if filtered_transcript == "hnh" and last_filtered_transcript == "w":
                    count["specific_words"]["whnh"] += 1
            if "hn/" in entry["transcript"]:  # also הנני counts as 'hnh'
                count["specific_words"]["hnh"] += 1
            if "om/" in entry["transcript"] and parsed_morph["sp"] == "ptcl":
                count["specific_words"]["oM"] += 1
    scores = []
    for feature in feature_list:
        feature_obj = Feature(feature[0], feature[1], feature[2], feature[3])
        nominator = sum_entries(count, feature_obj.numerator_keys)
        denominator = sum_entries(count, feature_obj.denominator_keys)
        scores.append(methods_name_dict[feature_obj.op](nominator, denominator))
    return scores


def get_starr_features(samples):
    morph_dict = get_morph_dict()
    features = [
        gen_sample_features(sample, morph_dict, feature_list) for sample in samples
    ]
    return np.array(features)


def get_starr_features_v2(samples) -> pd.DataFrame:
    morph_dict = get_morph_dict()
    features = [
        gen_sample_features(sample, morph_dict, feature_list) for sample in samples
    ]
    df = pd.DataFrame(features, columns=[f[0] for f in feature_list])
    return df
