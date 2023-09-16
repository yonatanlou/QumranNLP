from text_reader import read_text
from MorphParser import MorphParser
from os import path
from utils import root_path, filter_data_by_field, read_yaml
from collections import defaultdict, Counter
import re
import pandas as pd
import numpy as np
alphabet_chars = re.compile('[^a-zA-Z]')
from Starr.features_keys import sum_entries, Feature, feature_list, methods_name_dict


def check_pronoun(entry):
    parsed_morph = entry['parsed_morph']

    if parsed_morph['sp'] == 'subs' and 'sp2' in parsed_morph:
        return 'noun_bound_pronoun'
    if parsed_morph['sp'] == 'verb' and 'sp2' in parsed_morph:
        return 'verb_bound_pronoun'
    if parsed_morph['sp'] == 'ptcl':
        if parsed_morph['cl'] == 'rela':
            return 'relative_pronoun'
        if parsed_morph['cl'] == 'prep' and 'sp2' in parsed_morph:
            return 'preposition_bound_pronoun'
        if parsed_morph['cl'] == 'objm' and 'sp2' in parsed_morph:
            return 'object_marker_bound_pronoun'

    if parsed_morph['sp'] == 'pron':
        if parsed_morph['cl'] == 'indp':
            return 'independent_pronoun'
        if parsed_morph['cl'] == 'intr':
            return 'interrogative_pronoun'


text_file = path.join(root_path, 'Data', 'texts', 'abegg', 'dss_nonbib.txt')
yaml_dir = path.join(root_path, 'Data', 'yamls')
book_path = path.join(yaml_dir, 'non_sectarian_texts.yaml')
morph_parser = MorphParser(yaml_dir=yaml_dir)

pronoun_classes = ['noun_bound_pronoun', 'verb_bound_pronoun', 'preposition_bound_pronoun',
                   'object_marker_bound_pronoun', 'relative_pronoun',  'independent_pronoun', 'interrogative_pronoun']
general = ['verbs', 'nouns', 'words', 'particles', 'pronouns', 'adjectives']
specific_word_counts = ['ky', 'kya', 'hnh', 'whnh', 'aCr', 'oM']
morph_name_dict = {'noun_classes':{x: xx[1] for x, xx in morph_parser.raw_morph_yaml['values']['cl']['subs'].items()},
                    'verbal_stems':{x: xx[1] for x, xx in morph_parser.raw_morph_yaml['values']['vs'].items()},
                   'noun_states': {x: xx[1] for x, xx in morph_parser.raw_morph_yaml['values']['st'].items()},
                   'verbal_tense':{x: xx[1] for x, xx in morph_parser.raw_morph_yaml['values']['vt'].items()},
                   'verbal_mood': {x: xx[1] for x, xx in morph_parser.raw_morph_yaml['values']['md'].items()},
                   'adjective_states':{x: xx[1] for x, xx in morph_parser.raw_morph_yaml['values']['st'].items()},
                   'pronoun_classes': {x:x for x in pronoun_classes},
                   'particle_classes': {x: xx[1] for x, xx in morph_parser.raw_morph_yaml['values']['cl']['ptcl'].items()},
                   'specific_words': {x:x for x in specific_word_counts},
                   'general': {x:x for x in general}}

books_yaml = read_yaml(book_path)['nonbib']
books_flat = [scroll for scrolls in books_yaml for scroll in books_yaml[scrolls]]
data, lines = read_text(text_file, yaml_dir)
filtered_data = filter_data_by_field('scroll_name', books_flat, data)
for book in books_flat:
    for entry in filtered_data[book]:
        entry['parsed_morph'] = morph_parser.parse_morph(entry['morph'])

count = {book:{key:{k: 0 for k in morph_name_dict[key].values()} for key in morph_name_dict} for book in books_flat}
word_count_flag = False

for book in books_flat:
    last_filtered_transcript = ''
    for entry in filtered_data[book]:
        filtered_transcript = alphabet_chars.sub('', entry['transcript'])
        if entry['sub_word_num'] == '1':
            word_count_flag = False

        parsed_morph = entry['parsed_morph']
        if 'sp' in parsed_morph:
            if word_count_flag is False:
                count[book]['general']['words'] += 1
                word_count_flag = True
            if parsed_morph['sp'] == 'subs':
                count[book]['general']['nouns'] += 1
                if 'st' in parsed_morph and entry['parsed_morph']['st'] in morph_name_dict['noun_states']:
                    count[book]['noun_states'][morph_name_dict['noun_states'][entry['parsed_morph']['st']]] += 1
                if entry['parsed_morph']['cl'] in morph_name_dict['noun_classes'].keys():
                    count[book]['noun_classes'][morph_name_dict['noun_classes'][entry['parsed_morph']['cl']]] += 1

            if parsed_morph['sp'] == 'verb':
                count[book]['general']['verbs'] += 1
                if entry['parsed_morph']['vs'] in morph_name_dict['verbal_stems'].keys():
                    count[book]['verbal_stems'][morph_name_dict['verbal_stems'][entry['parsed_morph']['vs']]] += 1
                if 'vt' in parsed_morph and entry['parsed_morph']['vt'] in morph_name_dict['verbal_tense']:
                    count[book]['verbal_tense'][morph_name_dict['verbal_tense'][entry['parsed_morph']['vt']]] += 1
                    if 'participle' in morph_name_dict['verbal_tense'][entry['parsed_morph']['vt']]:
                        a = 1
                if 'md' in parsed_morph and entry['parsed_morph']['md'] in morph_name_dict['verbal_mood']:
                    count[book]['verbal_mood'][morph_name_dict['verbal_mood'][entry['parsed_morph']['md']]] += 1
            if parsed_morph['sp'] == 'ptcl':
                count[book]['general']['particles'] += 1
                count[book]['particle_classes'][morph_name_dict['particle_classes'][entry['parsed_morph']['cl']]] += 1

            if parsed_morph['sp'] == 'adjv':
                count[book]['general']['adjectives'] += 1
                if 'st' in parsed_morph:
                    count[book]['adjective_states'][morph_name_dict['adjective_states'][entry['parsed_morph']['st']]] += 1


            # count pronouns
            pronoun_type = check_pronoun(entry)
            if pronoun_type:
                count[book]['general']['pronouns'] += 1
                count[book]['pronoun_classes'][pronoun_type] += 1

            if filtered_transcript in morph_name_dict['specific_words']:
                if filtered_transcript not in ['om','oM','aCr']:
                    count[book]['specific_words'][filtered_transcript] += 1
                else:
                    if parsed_morph['sp'] == 'ptcl':
                        filtered_transcript = 'oM' if filtered_transcript == 'om' else filtered_transcript
                        count[book]['specific_words'][filtered_transcript] += 1

                if filtered_transcript == 'hnh' and last_filtered_transcript == 'w':
                    count[book]['specific_words']['whnh'] += 1
            if 'hn/' in entry['transcript']: # also הנני counts as 'hnh'
                count[book]['specific_words']['hnh'] += 1
            if 'om/' in entry['transcript'] and parsed_morph['sp'] == 'ptcl':
                count[book]['specific_words']['oM'] += 1

            last_entry = entry

total_counts = {}
for book, scrolls in books_yaml.items():
    total_counts[book] = defaultdict(Counter)
    for scroll in scrolls:
        for key in count[scroll].keys():
            total_counts[book][key].update(Counter(count[scroll][key]))

scores = {book: [] for book in books_yaml.keys()}
feature_names = []
for feature in feature_list:
    for book in books_yaml.keys():
        feature_obj = Feature(feature[0], feature[1], feature[2], feature[3])
        nominator = sum_entries(total_counts[book], feature_obj.numerator_keys)
        denominator = sum_entries(total_counts[book], feature_obj.denominator_keys)
        scores[book].append(methods_name_dict[feature_obj.op](nominator, denominator))


feature_names = [a[0] for a in feature_list]

score_table = np.array([scores[book] for book in books_yaml.keys()]).T
# score_table[-1, :] = score_table[-1, :] /100
feature_df = pd.DataFrame(score_table, index=feature_names,  columns=[book for book in books_yaml.keys()]).round(3)
feature_df.to_csv('starr_non_sectarian_features.csv')
