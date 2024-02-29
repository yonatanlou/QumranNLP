from functools import reduce
import operator

class Feature:
    def __init__(self, name, numerator_keys = None, denominator_keys = None, op=None):
        self.name = name
        self.numerator_keys = numerator_keys
        self.denominator_keys = denominator_keys
        self.op = op


def get_key(dictionary, key_list):
    return reduce(operator.getitem, key_list, dictionary)


def sum_key_values(dictionary):
    sum = 0
    if not isinstance(dictionary, dict):
        return dictionary
    for key, value in dictionary.items():
        sum += sum_key_values(dictionary[key])
    return sum


def sum_entries(counts, keys):
    if keys == None:
        return 1
    sum = 0
    for key in keys:
        sum += sum_key_values(get_key(counts, key))
    return sum


def ratio(x, y):
    if y == 0:
        return -1
    return x/y


def percentage(x, y):
    return (x/y)*100


methods_name_dict = {'ratio':ratio, 'percentage':percentage}


passive_stems = ['passive qal', 'nifal', 'pual', 'hofal', 'polal', 'pulal', 'poal', 'hotpaal', 'peal']
active_stems = ['piel', 'hifil', 'hitpael', 'palel', 'pilpel', 'polel', 'poel', 'tifil', 'hishtafel', 'hit-opel',
                'hitpalpel', 'nitpael', 'hithpolel', 'hishtaphel', 'pael', 'hitpoel']
unknown = ['hpealal', 'aphel', 'haphel', 'hophal', 'hithaphel', 'hithpeel', 'ishtaphel', 'ithpaal', 'ithpeel', 'peil',
           'shaphel', 'hithpaal', 'ithpoel', 'apoel']

feature_list = [
    ['construct to absolute nouns ratio', [['noun_states','construct']], [['noun_states','absolute']],'ratio'],
    ['construct nouns and adjectives percentage', [['noun_states','construct'], ['adjective_states','construct']], [['general', 'words']],'ratio'],
    ['noun to verb ratio', [['general', 'nouns']], [['general', 'verbs']],'ratio'],
    # ['independent CP precenteage', [['pronoun_classes','independent_pronoun'],['pronoun_classes','interrogative_pronoun']], [['general', 'words']], 'percentage'],
    ['definite_article_percentage', [['particle_classes', 'article'],['particle_classes', 'article + preposition']], [['general', 'words']], 'percentage'],
    # ['proper noun percentage', [['noun_classes', 'proper']], [['general', 'words']], 'percentage'],
    ['direct object marker percentage', [['particle_classes', 'object marker']], [['general', 'words']], 'percentage'],
    ['pronouns bound to nouns or verbs percentage', [['pronoun_classes', 'noun_bound_pronoun'], ['pronoun_classes', 'verb_bound_pronoun']], [['general', 'words']], 'percentage'],
    ['persuasive verb forms (imperative, jussive, cohorative) percentage', [['verbal_mood', 'cohortative'],['verbal_mood', 'jussive'], ['verbal_tense', 'imperative']], [['general', 'words']], 'percentage'],
    ['preterite percentage', [['verbal_tense', 'wayyiqtol aka preterite']], [['general', 'words']], 'percentage'],
    ['ky percentage', [['specific_words', 'ky']], [['general', 'words']], 'percentage'],
    ['aCr percentage', [['specific_words', 'aCr']], [['general', 'words']], 'percentage'],
    ['oM percentage', [['specific_words', 'oM']], [['general', 'words']], 'percentage'],
    ['kya percentage', [['specific_words', 'kya']], [['general', 'words']], 'percentage'],
    # ['hnh percentage', [['specific_words', 'hnh']], [['general', 'words']], 'percentage'],
    ['all conjunctions percentage', [['particle_classes', 'conjunction']], [['general', 'words']], 'percentage'],
    ['non-finite to finite verbs ratio',
     [['verbal_tense', x] for x in ['infinitive construct', 'infinitive absolute', 'participle', 'passive participle']],
     [['verbal_tense', x] for x in ['perfect', 'imperfect', 'wayyiqtol aka preterite', 'imperative']],
     'ratio'],
    ['passive verb forms percentage',
     [['verbal_stems', x] for x in passive_stems], [['general', 'words']], 'percentage'],
    ['total word count', [['general','words']], None, 'ratio']
]

