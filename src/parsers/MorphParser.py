from config import BASE_DIR
from utils import read_yaml
from os import path
import collections


class FieldNames:
    def __init__(self):
        self.source_line_num = 'source_line_num'
        self.frag_label = 'frag_label'
        self.frag_line_num = 'frag_line_num'
        self.word_line_num = 'word_line_num'
        self.sub_word_num = 'sub_word_num'
        self.sub_num = 'sub_num' # not sure what this is
        self.word_prefix = 'word_prefix'
        self.scroll_name = 'scroll_name'
        self.book_name = 'book_name' # only bib
        self.chapter_name = 'chapter_name' # only bib
        self.verse = 'verse'
        self.hverse = 'half_verse'
        self.interlinear = 'interlinear'
        self.script_type = 'script_type'
        self.transcript = 'transcript'
        self.lang = 'lang'
        self.lex = 'lex'
        self.morph = 'morph'
        self.unknown = 'unknown'
        self.null = '0'
        self.merr = 'merr'


class MorphParser:
    names = FieldNames()

    def __init__(self, yaml_dir=f"{BASE_DIR}/data/yamls"):
        self.raw_morph_yaml, self.morph_dict, self.speech_parts_values = self.read_morph_dict(yaml_dir)
        self.parsed_morphs = collections.defaultdict(set)
        self.names = FieldNames()

    def parse_morph(self, morphs):
        if isinstance(morphs, str):
            morphs = [morphs]

        parsed = None

        for morph in morphs:
            if morph in self.parsed_morphs:
                parsed = self.parsed_morphs[morph]
            else:
                parsed = self.__read_tag__(morph)
                self.parsed_morphs[morph] = parsed
        return parsed

    def __read_tag__(self, morph):
        tag = self.replace_esc(morph)
        parsed = {}
        part_num = 0

        while tag:
            m = tag[0]
            if part_num == 0 or tag.startswith("X"):
                tag, parsed = self.__read_tag_part__(tag, part_num, parsed)
                part_num += 1
            else:
                parsed.setdefault(self.names.merr, "")
                parsed[self.names.merr] += m
                tag = tag[1:]
        return parsed

    def __read_tag_part__(self, tag, part_n, tag_parse_dict):
        m = tag[0]
        tag = tag[1:]
        if not tag:
            return tag, {}
        pos = self.names.unknown if m == self.names.null else self.speech_parts_values.get(m, None)

        if not pos:
            tag_parse_dict.setdefault(self.names.merr, "")
            tag_parse_dict[self.names.merr] += m
            return tag, {}

        pos_field = 'sp' if not part_n else 'sp' + str(part_n+1)
        tag_parse_dict[pos_field] = pos

        features = self.morph_dict[pos].keys()
        for feature in features:
            if not tag:
                break
            m = tag[0]
            values = self.morph_dict[pos][feature]
            mft = f"{feature}{part_n + 1}" if part_n else feature

            value = self.names.unknown if m == self.names.null else values.get(m, None)
            if value is not None:
                tag_parse_dict[mft] = value
                tag = tag[1:]

        return tag, tag_parse_dict

    @staticmethod
    def read_morph_dict(yaml_dir):
        raw_morph_yaml = read_yaml(path.join(yaml_dir, 'morph.yaml'))
        value_dict = {}
        for (pos, feats) in raw_morph_yaml["tags"].items():
            pos_values = {}
            for feature_data in feats:
                feat_values = {}
                for (feature, values) in feature_data.items():
                    for v in values:
                        m = (
                            raw_morph_yaml["values"][feature][v][0]
                            if feature != 'cl'
                            else raw_morph_yaml["values"][feature][pos][v][0]
                        )
                        feat_values[m] = v
                pos_values[feature] = feat_values
            value_dict[pos] = pos_values
        speech_part_dict = {v[0]: k for (k, v) in raw_morph_yaml["values"]['sp'].items()}
        return raw_morph_yaml, value_dict, speech_part_dict

    def replace_esc(self, tag):
        for x in self.raw_morph_yaml['escapes']:
            tag = tag.replace(*x)
        return tag
