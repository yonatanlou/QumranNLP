import csv

import cp_statistics
from config import BASE_DIR
from parser_data import get_dss_data
from collections import Counter
from scipy.stats import hypergeom
from utils import Transcriptor

RESULT_PATH = "Results/CP/P_VALUE"

BIBLE_LIST = ['Deut', 'Prov', 'Ps']
SCROLLS_LIST = ['4Q417', '4Q418', '4Q423', '1Q26', '1QHa', '4Q418a', '4Q416', '4Q418b', '4Q415', '1QS',
                '4Q418c', '1Q35', '1QSa']
transcriptor = Transcriptor(f"{BASE_DIR}/data/yamls/heb_transcript.yaml")


def get_p_value(k, N, D, n):
    return 1 - hypergeom.cdf(k, N, D, n)


def save_words(name, words):
    with open(f"{RESULT_PATH}/{name}.txt", 'w') as f:
        latin_words = list(set([''.join(filter(str.isalnum, x['transcript'])) for x in words]))
        # latin_words = [l.replace('Ø', '') for l in latin_words]
        # latin_words = list(set([''.join([xx for xx in x['transcript'] if xx != 'Ø']) for x in latin_words]))
        heb_words = [transcriptor.latin_to_heb(x.replace('Ø', '')) for x in latin_words]
        f.writelines(', '.join(heb_words))
        print(heb_words)


def parser_book():
    data = [x for b in get_dss_data(SCROLLS_LIST, "nonbib").values() for x in b]
    total_words = len([x for x in data if x['sub_word_num'] == '1' and ''.join(filter(str.isalnum, x['transcript'])) != ''])  # N
    # data_bible = [x for b in get_dss_data(BIBLE_LIST, "bib") for x in b]
    map_list, properties_list = cp_statistics.import_data(date='070823')
    filtered_map_list = [x for x in map_list.values() if x['book'] not in BIBLE_LIST]
    num_of_cp_words = sum([len(x['rectums']) for x in filtered_map_list]) + len(filtered_map_list)  # D
    filtered_properties = [properties_list[i] for i in map_list if map_list[i]['book'] not in BIBLE_LIST and i != 4297]
    aggr_properties = {'Nog': len([x['N'] for x in filtered_properties if x['NoGRec'] != 0] +
                                   [x['N'] for x in filtered_properties if x['NoGReg'] != 0]),
                       'Quant': len([x['N'] for x in filtered_properties if x['Quantmid'] != 0] +
                                   [x['N'] for x in filtered_properties if x['Quantpre'] != 0]),
                       'pron': len([x['N'] for x in filtered_properties if x['Affpron'] != 0]),
                       'conj': len([x['N'] for x in filtered_properties if x['Affconj'] != 0]),
                       'prep': len([x['N'] for x in filtered_properties if x['Affprep'] != 0]),
                       'partpas': len([x['N'] for x in filtered_properties if x['RecpartPas'] != 0] +
                                   [x['N'] for x in filtered_properties if x['RegpartPas'] != 0]),
                       'partpf': len([x['N'] for x in filtered_properties if x['Recpartpf'] != 0] +
                                   [x['N'] for x in filtered_properties if x['Regpartpf'] != 0]),
                       'partpm': len([x['N'] for x in filtered_properties if x['Recpartpm'] != 0] +
                                     [x['N'] for x in filtered_properties if x['Regpm'] != 0]),
                       'partsf': len([x['N'] for x in filtered_properties if x['Recpartsf'] != 0] +
                                     [x['N'] for x in filtered_properties if x['Regsf'] != 0]),
                       'partsm': len([x['N'] for x in filtered_properties if x['Recpartsm'] != 0] +
                                     [x['N'] for x in filtered_properties if x['Regsm'] != 0]),

                       }
    with open(f"{RESULT_PATH}/p_value_results.csv", 'w', newline='') as f:
        csvwriter = csv.writer(f, delimiter='\t')
        csvwriter.writerow(["Category", "Total_number", "Number_in_PC", "p_value", "expected"])
        # god name:
        total_god_name = get_god_name(data)  # n
        p_value = get_p_value(k=aggr_properties['Nog'] - 1, N=total_words, D=num_of_cp_words, n=len(total_god_name))
        expected = num_of_cp_words * len(total_god_name) / total_words
        csvwriter.writerow(["God name", len(total_god_name), aggr_properties['Nog'], p_value, expected])
        save_words('Nog', total_god_name)
        # quant_counter
        quant_num = quant_counter(data)
        expected = num_of_cp_words * len(quant_num) / total_words
        p_value = get_p_value(k=aggr_properties['Quant'] - 1, N=total_words, D=num_of_cp_words, n=len(quant_num))
        csvwriter.writerow(["Quant", len(quant_num), aggr_properties['Quant'], p_value, expected])
        save_words('Quant', quant_num)

        # conj_counter
        conj_num = get_conj(data)
        expected = num_of_cp_words * len(conj_num) / total_words
        p_value = get_p_value(k=aggr_properties['conj'] - 1, N=total_words, D=num_of_cp_words, n=len(conj_num))
        csvwriter.writerow(["Conj", len(conj_num), aggr_properties['conj'], p_value, expected])
        save_words('Conj', conj_num)

        # prep
        prep_num = get_prep(data)
        expected = num_of_cp_words * len(prep_num) / total_words
        p_value = get_p_value(k=aggr_properties['prep'] - 1, N=total_words, D=num_of_cp_words, n=len(prep_num))
        csvwriter.writerow(["Prep", len(prep_num), aggr_properties['prep'], p_value, expected])
        save_words('prep', prep_num)

        # partpas
        part_num = part_filtered(data, passive=True)
        expected = num_of_cp_words * len(part_num) / total_words
        p_value = get_p_value(k=aggr_properties['partpas'] - 1, N=total_words, D=num_of_cp_words, n=len(part_num))
        csvwriter.writerow(["passive", len(part_num), aggr_properties['partpas'], p_value, expected])
        save_words('passive', part_num)

        # partpf
        part_num = part_filtered(data, feminine=True, singular=False)
        expected = num_of_cp_words * len(part_num) / total_words
        p_value = get_p_value(k=aggr_properties['partpf'] - 1, N=total_words, D=num_of_cp_words, n=len(part_num))
        csvwriter.writerow(["pf", len(part_num), aggr_properties['partpf'], p_value, expected])
        save_words('pf', part_num)

        # partpm
        part_num = part_filtered(data, feminine=False, singular=False)
        expected = num_of_cp_words * len(part_num) / total_words
        p_value = get_p_value(k=aggr_properties['partpm'] - 1, N=total_words, D=num_of_cp_words, n=len(part_num))
        csvwriter.writerow(["pm", len(part_num), aggr_properties['partpm'], p_value, expected])
        save_words('pm', part_num)

        # partsf
        part_num = part_filtered(data, feminine=True, singular=True)
        expected = num_of_cp_words * len(part_num) / total_words
        p_value = get_p_value(k=aggr_properties['partsf'] - 1, N=total_words, D=num_of_cp_words, n=len(part_num))
        csvwriter.writerow(["sf", len(part_num), aggr_properties['partsf'], p_value, expected])
        save_words('sf', part_num)

        # partsm
        part_num = part_filtered(data, feminine=False, singular=True)
        expected = num_of_cp_words * len(part_num) / total_words
        p_value = get_p_value(k=aggr_properties['partsm'] - 1, N=total_words, D=num_of_cp_words, n=len(part_num))
        csvwriter.writerow(["sm", len(part_num), aggr_properties['partsm'], p_value, expected])
        save_words('sm', part_num)


def part_filtered(data, feminine=False, singular=False, passive=False):
    if passive:
        return get_passive(data)
    part = get_part(data)
    gender_filtered = get_gender(part, feminine)
    return get_number(gender_filtered, singular)


def quant_counter(data):
    # כמתים: מספרים, רוב, כלֿ
    number = [x for x in data if x['parsed_morph'].get('sp', '') == 'numr']
    kl = [x for x in data if x['transcript'] == 'kl'] + [x for x in data if x['transcript'] == 'kwl']
    rov = [x for x in data if x['lex'] == 'rOb']
    return number + kl + rov


def get_prep(data):
    # מילות יחס:
    prep = [x for x in data if x['parsed_morph'].get('cl') == 'prep']
    artp = [x for x in data if x['parsed_morph'].get('cl') == 'artp']
    # asher = [x for x in data if x['parsed_morph'].get('cl') == 'rela']
    return prep + artp


def get_conj(data):
    # מילות קישור:
    return [x for x in data if x['parsed_morph'].get('cl') == 'conj']


def get_god_name(data):
    yhwh = [x for x in data if ''.join(filter(str.isalnum, x['transcript'])) == "yhwh"]
    al = [x for x in data if ''.join(filter(str.isalnum, x['transcript']))[:2] == "al" and x["parsed_morph"].get('sp') == 'subs'
          and x["parsed_morph"].get('gn') == 'm']
    al = [x for x in al if len(''.join(filter(str.isalnum, x['transcript']))) == 2
          or ''.join(filter(str.isalnum, x['transcript']))[2] in ["y", "h", "w"]]
    yh = [x for x in data if ''.join(filter(str.isalnum, x['transcript'])) == "yh" and
          x["parsed_morph"].get('sp') == 'subs']
    olywn = [x for x in data if ''.join(filter(str.isalnum, x['transcript'])) == "olywn"
             and x["parsed_morph"].get('sp') == 'subs']
    kadosh = [x for x in data if ''.join(filter(str.isalnum, x['transcript'])) == "qdwC"
             and x["parsed_morph"].get('sp') == 'subs']
    # collections.Counter([''.join(filter(str.isalnum, x['transcript'])) for b in filtered_data.values() for x in b if
    #                      x['morph'][0] == 'D'])
    return yhwh + al + yh + olywn + kadosh


def get_number(data, singular=True):
    # יחיד/ רבים
    if singular:
        return [x for x in data if x['parsed_morph'].get('nu') == 's']
    else:
        return [x for x in data if x['parsed_morph'].get('nu') == 'p']


def get_gender(data, feminine=True):  # Masculine m
    # זכר/ נקבה
    if feminine:
        return [x for x in data if x['parsed_morph'].get('gn') == 'f']
    else:
        return [x for x in data if x['parsed_morph'].get('gn') == 'm']


def get_passive(data):
    # סביל
    return [x for x in data if x['parsed_morph'].get('vt', '') == 'ptcp'
            or (x['parsed_morph'].get('vt', '') == 'ptca' and x['parsed_morph'].get('vs') in ['passive', 'pual', 'nifal'])]
    # return [x for x in data if x['parsed_morph'].get('vt', '') == 'ptcp']


def get_part(data):
    return [x for x in data if x['parsed_morph'].get('vt', '') == 'ptca']


parser_book()
