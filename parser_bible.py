import collections

from MorphParser import FieldNames, MorphParser

DATA = 'Data/texts/open_scriptures/bible.txt'
line_fields_names = FieldNames()
yaml_dir = "Data/yamls"


def get_verbs_categories(filtered_data):
    tmp = [x for b in filtered_data.values() for x in b]
    first_type_v = [x for x in tmp if x['morph'][0] == 'V' and len(x['morph'].split('/')[0]) == 6]
    options = collections.defaultdict(list)
    [options[x["morph"][1:3]].append(x['transcript']) for x in first_type_v]
    second_type_v = [x for x in tmp if x['morph'][0] == 'V' and len(x['morph'].split('/')[0]) == 3]
    [options[x["morph"][1]].append(x['transcript']) for x in second_type_v]
    for k, x in options.items():
        with open(f"Results/CP/P_VALUE/bible_verbs/{k}.txt", 'w') as f:
            f.write(', '.join(list(set(x))))


def parser_bible(books_list):
    filtered_data = collections.defaultdict(list)

    with open(DATA) as f:
        for r in f.readlines():
            l = r.replace('\n', '').split('\t')
            if l[0] not in books_list:
                continue
            chapter_name, a = l[1].split(':')
            verse, word_num = a.split('.')
            words_idx = l[4].split('/')
            words = l[2].split('/')
            morph = l[3][1:].split('/')

            for i, _ in enumerate(words_idx):
                parsed_word = collections.defaultdict(lambda: "")
                parsed_word[line_fields_names.book_name] = l[0]
                parsed_word[line_fields_names.chapter_name] = chapter_name
                parsed_word[line_fields_names.verse] = verse
                word_line_num = word_num if len(words_idx) == 1 else f"{word_num}.{i + 1}"
                parsed_word[line_fields_names.word_line_num] = word_line_num
                parsed_word["word_id"] = words_idx[i]
                if len(words_idx) > 1:
                    parsed_word[line_fields_names.transcript] = words[i]
                    parsed_word[line_fields_names.morph] = morph[i]
                else:
                    parsed_word[line_fields_names.transcript] = l[2]
                    parsed_word[line_fields_names.morph] = l[3][1:]
                filtered_data[parsed_word[line_fields_names.book_name]].append(parsed_word)

    get_verbs_categories(filtered_data)
    return
    morph_parser = MorphParser(yaml_dir=yaml_dir)
    a = collections.defaultdict(list)
    for book in books_list:
        for entry in filtered_data[book]:
            entry['morph'] = parse_morph(entry)

    return filtered_data


def parse_morph(entry):
    morph = {}
    tags = entry['morph'].split('/')

    tag = tags[0]
    if tag[0] == 'N':
        assert len(tag) in [2, 5]
        morph['sp'] = 'n'
        assert tag[1] in ['p', 'c', 'g']
        if tag[1] == 'p':
            morph['cl'] = 'p'
            return morph
        morph['cl'] = tag[1]
        assert tag[2] in ['m', 'f', 'b']
        morph['g'] = tag[2]
        assert tag[3] in ['s', 'p', 'd']
        morph['nu'] = tag[3]
        assert tag[4] in ['a', 'c']
        morph['st'] = tag[4]
    elif tag[0] == 'V':
        assert len(tag) in [3, 6]
        morph['sp'] = 'v'
        if len(tag) == 3:
            # assert tag[1] in ['h', 'q', 'N', 'h', 't', 'r', 'p', 'l', 'z', 'o', 'P']
            morph['vs'] = tag[1]
            assert tag[2] in ['a', 'c']
            morph['st'] = tag[2]
        else:
            if tag[3] in ['1', '2', '3']:
                assert tag[1] in ['p', 'q', 'h', 'N', 't', 'v', 'H', 'r', 'P', 'D']
                assert tag[2] in ['p', 'w', 'v', 'i', 'j', 'q', 'h']
                morph['ps'] = tag[3]
                assert tag[4] in ['m', 'f', 'c']
                morph['gn'] = tag[4]
                assert tag[5] in ['s', 'p']
                morph['nu'] = tag[5]
            else:
                print(entry["transcript"])
                assert tag[2] in ['r', 's']
                assert tag[3] in ['m', 'f']
                morph['gn'] = tag[3]
                assert tag[4] in ['s', 'p']
                morph['nu'] = tag[4]
                assert tag[5] in ['a', 'c']
                morph['st'] = tag[5]

            # assert tag[5] in ['s', 'a', 'p', 'c']
            # morph['nu'] = tag[5]

            # print(tag[3])
            # assert tag[3] in ['1', '2', '3']
            # morph['ps'] = tag[3]

    elif tag[0] == 'A':
        morph['sp'] = 'v'
        morph['cl'] = tag[1]

    elif tag[0] == 'P':
        morph['sp'] = 'p'
    elif tag[0] == 'R':
        morph['sp'] = ''
        morph['cl'] = 'prep'
    return morph

    adjv: [a, adjective]
    numr: [u, numeral]
    subs: [n, noun]
    ptcl: [P, particle]
    pron: [p, CP]
    verb: [v, verb]
    suff: [X, suffix]

    return tags[0]


def relative(data):
    [x for b in data.values() for x in b if x['morph'][0] == 'R']
# C
# S
# D

def pesron(data):
    collections.Counter([x['transcript'] for b in filtered_data.values() for x in b if x['morph'][0] == 'P'])


parser_bible(books_list=['Deut', 'Prov', 'Ps'])