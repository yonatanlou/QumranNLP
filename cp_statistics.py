import csv
import pandas as pd
from collections import Counter, defaultdict
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import parser_data
from config import BASE_DIR


def import_data(date):
    map_list = {}
    properties_list = {}
    all_book = []
    book_name = {"Psamls": "Ps", "Psaims": "Ps", "Psalms": "Ps", "Psalms1": "Ps", '4Q415': 'Musar_Lamevin',
                 '1QS': '1QS', '1QSa': '1QSa', '1Q26': 'Musar_Lamevin', '4Q418b': 'Musar_Lamevin', '4Q423': 'Musar_Lamevin',
                 '4Q418': 'Musar_Lamevin',
                 '4Q416': "Musar_Lamevin", '4Q417': "Musar_Lamevin", '4Q418c': 'Musar_Lamevin', '1Q35': "hodayot",
                 '4Q418a': 'Musar_Lamevin',
                 'Deu': 'Deut', '1QH': 'Hodayot', 'Prov': 'Prov'}
    with open(f'{BASE_DIR}/data/CP/{date}/map.csv', newline='') as f:
        data = csv.reader(f)
        for row in data:
            if not row[-1].replace(" ", "").isnumeric():
                continue
            book = row[2].encode("ascii", "ignore").decode().strip().replace(" ", ":").split(":")[0]
            book = book_name[book]

            all_book.append(book)
            row_dict = {"counter": int(row[-1].replace(" ", "")), "ref": row[2].strip(), "CP": row[5].strip(),
                        "book": book,
                        "regen": row[11].replace(" ", ""),
                        "rectums": [row[i].replace(" ", "") for i in range(10, 5, -1) if row[i].replace(" ", "") != '']}
            map_list[int(row[-1].replace(" ", ""))] = row_dict

    with open(f'data/CP/{date}/properties.csv', newline='') as f:
        df = pd.read_csv(f)
        df.drop(df.columns[[1, 2, 3, 26, 27]], axis=1, inplace=True)
        for index, row in df.iterrows():
            row_dict = {c.replace(" ", ""): get_val(str(row[c]), row[0], c) for c in df.columns}
            properties_list[row[0]] = row_dict
    print(set(all_book))
    return map_list, properties_list
    # for i in range(2*len(properties_list)):
    #     if any(x['counter'] == i for x in map_list) == any(x['N'] == i for x in properties_list):
    #         continue
    #     print(f"wrong {i}")


def get_val(x, row, c):
    c = c.replace(" ", "")
    if '?' in x:
        x = x.replace("(", "").replace(")", "").replace("?", "")
    x = x.replace(" ", "")
    if x == 'nan' or x == '':
        return 0
    if x == 'v' or x == 'V':
        return 1
    if c in ['NoGRec', 'NoGReg', 'NoGCPh']:
        return x
    if c == 'Affprep':
        return 1
    if x.isnumeric():
        return int(x)
    print("x= ", x, row, c)


def get_number_of_words():
    bible = ['Deut', 'Prov', 'Ps']#['Psalms', 'Prov', 'Deu']
    scrolls = ['4Q417', '4Q418', '4Q423', '1Q26', '1QHa', '4Q418a', '4Q416', '4Q418b', '4Q415', '1QS', '4Q418c', '1Q35',
               '1QSa']
    data = parser_data.get_dss_data(scrolls)
    books_words_counter = {}
    for book_name, book_data in data.items():
        samples = gen_sample(book_data)
        books_words_counter[book_name] = len(samples)
    data = parser_data.get_dss_data(bible, "bib")
    for book_name, book_data in data.items():
        samples = gen_sample(book_data)
        books_words_counter[book_name] = len(samples)
    return books_words_counter


def get_statistic_for_group(map_list, properties_list, idx, normalized_factor=1):
    properties_to_check = ['NoGCPh', 'NoGRec', 'NoGReg', 'Quantmid', 'Quantpre', 'Affpron', 'Affconj', 'Affprep',
                           'MultRec', 'Multregs', 'RecpartPas', 'Recpartpf', 'Recpartpm', 'Recpartsf', 'Recpartsm',
                           'RegpartPas', 'Regpartpf', 'Regpm', 'Regsf', 'Regsm']
    count = len(idx)
    print(count)
    properties_statistic = {'total_words': normalized_factor, 'construct_count': count, 'percent': count / normalized_factor}
    for p in properties_to_check:
        cells = [val[p] for k, val in properties_list.items() if k in idx]
        non_empty_cells = [c for c in cells if c != 0 and c is not None]
        if p in ['MultRec', 'Multregs']:
            c = Counter(non_empty_cells)
            if len(c.keys()) == 0:
                properties_statistic[p] = []
            else:
                properties_statistic[p] = [c[k] for k in range(max(c.keys()) + 1)]
            continue
        if p in ['NoGCPh', 'NoGRec', 'NoGReg']:
            properties_statistic[p] = len(non_empty_cells) / normalized_factor
        else:
            properties_statistic[p] = np.sum(non_empty_cells) / normalized_factor
    regen = Counter([x['regen'] for i, x in map_list.items() if i in idx])
    rectum = Counter([y for i, x in map_list.items() if i in idx for y in x['rectums']])

    properties_statistic['unique_regen'] = len(regen.keys())
    properties_statistic['total_regen'] = sum(regen.values())
    properties_statistic['unique_rectum'] = len(rectum.keys())
    properties_statistic['total_rectum'] = sum(rectum.values())
    properties_statistic['regen_dict'] = regen
    properties_statistic['rectum_dict'] = rectum
    return properties_statistic


def create_graphs(first_statistic, second_statistic, first_name, second_name, date):
    fig, axs = plt.subplots(ncols=2, sharex='col', sharey='row')
    fig.suptitle(f'Mult-Rec: {first_name} v.s {second_name}')
    first_statistic['MultRec'][1] = 0
    first_statistic['Multregs'][1] = 0
    second_statistic['MultRec'][1] = 0
    second_statistic['Multregs'][1] = 0
    axs[0].bar([i for i in range(len(first_statistic['MultRec']))], first_statistic['MultRec'])
    axs[1].bar([i for i in range(len(second_statistic['MultRec']))], second_statistic['MultRec'])
    x_lim = max(len(first_statistic['MultRec']), len(second_statistic['MultRec']))
    axs[0].set_xlim(-0.5, x_lim)
    axs[1].set_xlim(-0.5, x_lim)
    plt.savefig(f"Results/CP/{date}/Mult_Rec_{first_name}_{second_name}")

    fig, axs = plt.subplots(ncols=2, sharex='col', sharey='row')
    fig.suptitle(f'Mult-Regs:{first_name} v.s {second_name}')
    axs[0].bar([i for i in range(len(first_statistic['Multregs']))], first_statistic['Multregs'])
    axs[1].bar([i for i in range(len(second_statistic['Multregs']))], second_statistic['Multregs'])
    x_lim = max(len(first_statistic['Multregs']), len(second_statistic['Multregs']))
    axs[0].set_xlim(-0.5, x_lim)
    axs[1].set_xlim(-0.5, x_lim)
    plt.savefig(f"{BASE_DIR}/Results/CP/{date}/Mult_Regs_{first_name}_{second_name}")

    plt.figure(figsize=(15, 10))
    plt.title(f"30 most frequent regens {first_name}")
    word_counter = list(first_statistic['regen_dict'].items())
    word_counter = sorted(word_counter, reverse=True, key=lambda x: x[1])
    plt.bar([w[0] for w in word_counter[:30]], [w[1] / first_statistic['construct_count'] for w in word_counter[:30]])
    plt.xticks(rotation=50)
    plt.savefig(f"{BASE_DIR}/Results/CP/{date}/regen_{first_name}")

    plt.figure(figsize=(15, 10))
    plt.title(f"30 most frequent regen {second_name}")
    word_counter = list(second_statistic['regen_dict'].items())
    word_counter = sorted(word_counter, reverse=True, key=lambda x: x[1])
    plt.bar([w[0]for w in word_counter[:30]], [w[1] / second_statistic['construct_count'] for w in word_counter[:30]])
    plt.xticks(rotation=50)
    plt.savefig(f"{BASE_DIR}/Results/CP/{date}/regen_{second_name}")

    plt.figure(figsize=(15, 10))
    plt.title(f"30 most frequent rectums {first_name}")
    word_counter = list(first_statistic['rectum_dict'].items())
    word_counter = sorted(word_counter, reverse=True, key=lambda x: x[1])
    plt.bar([w[0] for w in word_counter[:30]], [w[1] / first_statistic['construct_count'] for w in word_counter[:30]])
    plt.xticks(rotation=50)
    plt.savefig(f"{BASE_DIR}/Results/CP/{date}/rectum_{first_name}")

    plt.figure(figsize=(15, 10))
    plt.title(f"30 most frequent rectums {second_name}")
    word_counter = list(second_statistic['rectum_dict'].items())
    word_counter = sorted(word_counter, reverse=True, key=lambda x: x[1])
    plt.bar([w[0] for w in word_counter[:30]], [w[1] / second_statistic['construct_count'] for w in word_counter[:30]])
    plt.xticks(rotation=50)
    plt.savefig(f"{BASE_DIR}/Results/CP/{date}/rectum_{second_name}")


def get_statistics(date):
    bible_books = ['Deut', 'Prov', 'Ps']  # ['Psalms', 'Prov', 'Deu']
    scrolls_books = ['4Q417', '4Q418', '4Q423', '1Q26', '1QHa', '4Q418a', '4Q416', '4Q418b', '4Q415',
                     '1QS', '4Q418c', '1Q35', '1QSa']
    properties = ['total_words', 'construct_count', 'percent', 'unique_regen', 'total_regen',  'unique_rectum',
                  'total_rectum', 'NoGCPh', 'NoGRec', 'NoGReg', 'Quantmid', 'Quantpre', 'Affpron', 'Affconj', 'Affprep',
                  'RecpartPas', 'Recpartpf', 'Recpartpm', 'Recpartsf', 'Recpartsm', 'RegpartPas', 'Regpartpf',
                  'Regpm', 'Regsf', 'Regsm']
    number_of_words = get_number_of_words()

    print("start")
    map_list, properties_list = import_data(date)

    ## statistic per book
    book_statistics = []
    for book in bible_books + scrolls_books:
        data = [i for i, val in map_list.items() if val['book'] == book]
        normalize = number_of_words[book]
        statistic = get_statistic_for_group(map_list, properties_list, data, normalize)
        statistic["book"] = book
        book_statistics.append(statistic)

    with open(f'{BASE_DIR}/Results/CP/{date}/books_statistic.csv', 'w') as f:
        fieldnames = ['book'] + properties
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for statistic in book_statistics:
            writer.writerow({k: v for k, v in statistic.items() if k in fieldnames})

    ## prose vs poem
    prose = [i for i, val in properties_list.items() if val['Pr'] == 1]
    poem = [i for i, val in properties_list.items() if val['Po'] == 1]
    normalize_prose = 1
    normalize_poem = 1
    prose_statistic = get_statistic_for_group(map_list, properties_list, prose, normalize_prose)
    poem_statistic = get_statistic_for_group(map_list, properties_list, poem, normalize_poem)
    create_graphs(prose_statistic, poem_statistic, "Prose", "Poem", date)

    ## bible v.s scrolls
    normalize_scrolls = sum([v for k, v in number_of_words.items() if k in scrolls_books])
    normalize_bible = sum([v for k, v in number_of_words.items() if k in bible_books])
    bible_refs = ['Psamls', 'Psaims', 'Psalms1', 'Psalms', 'Prov', 'Deu']
    bible = [i for i, val in map_list.items() if val['ref'].split(' ')[0].split(":")[0] in bible_refs]
    scroll = [i for i in map_list if i not in bible]
    bible_statistic = get_statistic_for_group(map_list, properties_list, bible, normalize_bible)
    scroll_statistic = get_statistic_for_group(map_list, properties_list, scroll, normalize_scrolls)
    create_graphs(bible_statistic, scroll_statistic, "Bible", "Scrolls", date)

    with open(f'{BASE_DIR}/Results/CP/{date}/prose_poem.csv', 'w') as f:
        fieldnames = ['property', 'prose', 'poem']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in properties:
            writer.writerow({'property': p, 'prose': prose_statistic[p], 'poem': poem_statistic[p]})

    with open(f'{BASE_DIR}/Results/CP/{date}/bible_scrolls.csv', 'w') as f:
        fieldnames = ['property', 'bible', 'scrolls']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in properties:
            writer.writerow({'property': p, 'bible': bible_statistic[p], 'scrolls': scroll_statistic[p]})


def tmp(map_list, properties_list, is_bible, date):
    bible_refs = ['Psamls', 'Psaims', 'Prov', 'Deu', 'Psalms1', 'Psalms']
    book = [i for i, val in map_list.items() if (val['ref'].split(' ')[0].split(":")[0] in bible_refs) == is_bible]
    prose = [i for i, val in properties_list.items() if val['Pr'] == 1 and i in book]
    prose_regen = Counter([x['regen'] for i, x in map_list.items() if i in prose])
    prose_regen_word_counter = list(prose_regen.items())
    prose_regen_word_counter = sorted(prose_regen_word_counter, reverse=True, key=lambda x: x[1])
    prose_rectum = Counter([y for i, x in map_list.items() if i in prose for y in x['rectums']])
    prose_rectum_word_counter = list(prose_rectum.items())
    prose_rectum_word_counter = sorted(prose_rectum_word_counter, reverse=True, key=lambda x: x[1])

    poem = [i for i, val in properties_list.items() if val['Po'] == 1 and i in book]
    poem_regen = Counter([x['regen'] for i, x in map_list.items() if i in poem])
    poem_regen_word_counter = list(poem_regen.items())
    poem_regen_word_counter = sorted(poem_regen_word_counter, reverse=True, key=lambda x: x[1])
    poem_rectum = Counter([y for i, x in map_list.items() if i in poem for y in x['rectums']])
    poem_rectum_word_counter = list(poem_rectum.items())
    poem_rectum_word_counter = sorted(poem_rectum_word_counter, reverse=True, key=lambda x: x[1])
    words_to_counter = prose_regen_word_counter[:10] + prose_rectum_word_counter[:10] + \
                       poem_regen_word_counter[:10] + poem_rectum_word_counter[:10]
    words_to_counter = list(set([w[0] for w in words_to_counter]))

    prose_regen_most_freq = [prose_regen[w] / len(book) for w in words_to_counter]
    prose_rectum_most_freq = [prose_rectum[w] / len(book) for w in words_to_counter]
    poem_regen_most_freq = [poem_regen[w] / len(book)for w in words_to_counter]
    poem_rectum_most_freq = [poem_rectum[w] / len(book) for w in words_to_counter]
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(12)

    ax.bar(words_to_counter, prose_regen_most_freq, label="regen: prose")
    ax.bar(words_to_counter, poem_regen_most_freq, bottom=prose_regen_most_freq, label="regen: poem")
    ax.bar(words_to_counter, prose_rectum_most_freq, bottom=np.add(prose_regen_most_freq, poem_regen_most_freq),
           label="rectum: prose")
    ax.bar(words_to_counter, poem_rectum_most_freq,
           bottom=np.add(np.add(prose_regen_most_freq, prose_rectum_most_freq), poem_regen_most_freq), label="rectum: poem")
    plt.xticks(rotation=50)
    plt.legend()
    plt.title(f"10 most frequency words from each category: {'bible' if is_bible else 'scrolls'}")
    plt.savefig(f"{BASE_DIR}/Results/CP/{date}/most_frequency_words_{'bible' if is_bible else 'scrolls'}")


def build_network(date):
    map_list, properties_list = import_data(date)
    bible_refs = ['Psamls', 'Psaims', 'Prov', 'Deu', 'Psalms1', 'Psalms']
    bible_idx = [i for i, val in map_list.items() if val['ref'].split(' ')[0].split(":")[0] in bible_refs]

    bible = [map_list[i] for i in map_list if i in bible_idx]
    regens = [p['regen'] for p in bible]
    rectums = [p['rectums'] for p in bible]
    G = nx.DiGraph()
    degrees_dict = defaultdict(int)
    edge_weights = defaultdict(int)
    edge_to_weight_for_graph = defaultdict(int)
    regen_count = defaultdict(int)
    total_count = defaultdict(int)

    for i, v in enumerate(regens):
        for u in rectums[i]:
            degrees_dict[v[::-1]] += 1
            degrees_dict[u[::-1]] += 1
            edge_weights[(f"{v[::-1]} (-) {u[::-1]}")] += 1
            edge_to_weight_for_graph[(v, u)] += 1
            regen_count[v] += 1
            regen_count[u] += 0
            total_count[v] += 1
            total_count[u] += 1
    for v in total_count:
        weight = regen_count[v] / total_count[v]
        G.add_node(v, size=weight)

    for edge, weight in edge_to_weight_for_graph.items():
        G.add_edge(edge[0], edge[1], weight=weight)

    with open(f'{BASE_DIR}/Results/CP/{date}/bible_node_list.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["node_name", "degree"])
        for k, v in degrees_dict.items():
            writer.writerow([k, v])

    with open(f'{BASE_DIR}/Results/CP/{date}/bible_words_statistic.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["node_name", "regen", "rectum", "regen/total"])
        for v in G.nodes:
            writer.writerow([v, G.out_degree[v], G.in_degree[v], G.out_degree[v] / (G.out_degree[v] + G.in_degree[v])])

    with open(f'{BASE_DIR}/Results/CP/{date}/bible_edge_list.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["edge_name", "weight"])
        for k, v in edge_weights.items():
            writer.writerow([k, v])
    nx.write_graphml_lxml(G, f"{BASE_DIR}/Results/CP/{date}/bible.graphml")
    nx.write_gexf(G, f"{BASE_DIR}/Results/CP/{date}/bible.gexf")

    print(len(degrees_dict), len(edge_weights))

    scroll = [map_list[i] for i in map_list if i not in bible_idx]
    regens = [p['regen'] for p in scroll]
    rectums = [p['rectums'] for p in scroll]
    G = nx.DiGraph()
    degrees_dict = defaultdict(int)
    edge_weights = defaultdict(int)
    edge_to_weight_for_graph = defaultdict(int)
    regen_count = defaultdict(int)
    total_count = defaultdict(int)

    for i, v in enumerate(regens):
        for u in rectums[i]:
            degrees_dict[v[::-1]] += 1
            degrees_dict[u[::-1]] += 1
            edge_weights[(f"{v[::-1]} (-) {u[::-1]}")] += 1
            edge_to_weight_for_graph[(v, u)] += 1
            regen_count[v] += 1
            regen_count[u] += 0
            total_count[v] += 1
            total_count[u] += 1

    for v in total_count:
        weight = regen_count[v] / total_count[v]
        G.add_node(v, size=weight)

    for edge, weight in edge_to_weight_for_graph.items():
        G.add_edge(edge[0], edge[1], weight=weight)

    with open(f'{BASE_DIR}/Results/CP/{date}/scrolls_node_list.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["node_name", "degree"])
        for k, v in degrees_dict.items():
            writer.writerow([k, v])

    with open(f'{BASE_DIR}/Results/CP/{date}/scrolls_words_statistic.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["node_name", "regen", "rectum", "regen/total"])
        for v in G.nodes:
            writer.writerow([v, G.out_degree[v], G.in_degree[v], G.out_degree[v] / (G.out_degree[v] + G.in_degree[v])])

    with open(f'{BASE_DIR}/Results/CP/{date}/scrolls_edge_list.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["edge_name", "weight"])
        for k, v in edge_weights.items():
            writer.writerow([k, v])

    nx.write_graphml_lxml(G, f"{BASE_DIR}/Results/CP/{date}/scrolls.graphml")
    nx.write_gexf(G, f"{BASE_DIR}/Results/CP/{date}/scrolls.gexf")

    print(len(degrees_dict), len(edge_weights))


def gen_sample(entries):
    all_words_entries = []
    curr_word_entries = []
    for e, entry in enumerate(entries):
        if 'sp' not in entry['parsed_morph']:
            curr_word_entries = []
            continue
        else:
            if e != len(entries)-1 and entries[e+1]['word_line_num'] == entries[e]['word_line_num']:
                curr_word_entries.append(entry)
                if entries[e+1]['transcript'] == '.':
                    curr_word_entries.append(entries[e+1])
            else:
                curr_word_entries.append(entry)
                if entries[e+1]['transcript'] == '.':
                    curr_word_entries.append(entries[e+1])
                all_words_entries.append(curr_word_entries)
                curr_word_entries = []

    return all_words_entries


