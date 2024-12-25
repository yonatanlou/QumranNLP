from tf.app import use

A = use("ETCBC/dss", hoist=globals())

from tqdm import tqdm
from collections import defaultdict


def get_morphological_features(word):
    """Extract morphological features for a given word node."""
    return {
        "sp": F.sp.v(word),
        "cl": F.cl.v(word),
        "ps": F.ps.v(word),
        "gn": F.gn.v(word),
        "nu": F.nu.v(word),
        "st": F.st.v(word),
        "vs": F.vs.v(word),
        "vt": F.vt.v(word),
        "md": F.md.v(word),
    }


def get_biblical_from_line(line):
    """
    Returns the biblical section of a line.
    """
    bib = F.biblical.v(line)
    if bib == None:
        return "nonbib"
    elif bib == 1:
        return "bib"
    elif bib == 2:
        return "biblical_non_biblical"


def get_biblical_info(word):
    """Get biblical information related to the word node."""
    return {
        "bib": str(get_biblical_from_line(word)),
        "bib_book": F.book.v(word),
        "bib_chapter": F.chapter.v(word),
    }


def get_scroll_and_chapter_info(word):
    """Get scroll and chapter information for a given word node."""
    scroll_and_chapter = A.sectionStrFromNode(word)
    scroll, chapter_info = scroll_and_chapter.split(" ")
    frag_label, frag_line_num = chapter_info.split(":")
    return scroll_and_chapter, scroll, frag_label, frag_line_num


def process_word(scroll_node, word_line_num, sub_word_num):
    filtered_data = defaultdict(list)
    for word in L.d(scroll_node, otype="word"):
        (
            scroll_and_chapter,
            scroll,
            frag_label,
            frag_line_num,
        ) = get_scroll_and_chapter_info(word)
        transcript = T.text(word)
        lexeme = F.glex.v(word)
        morphological_features = get_morphological_features(word)
        biblical_info = get_biblical_info(word)
        lang = F.lang.v(word)
        srcLn = F.srcLn.v(word)
        word_type = F.type.v(word)
        after = F.after.v(word)

        word_entry = {
            "frag_label": frag_label,
            "frag_line_num": frag_line_num,
            "word_line_num": str(word_line_num),
            "sub_word_num": str(sub_word_num),
            "book_and_chapter": scroll_and_chapter,
            "scroll_name": scroll,
            "transcript": transcript,
            "lex": lexeme,
            "parsed_morph": morphological_features,
            "lang": lang,
            "srcLn": srcLn,
            "type_of": word_type,
            "after": after,
        }
        word_entry.update(biblical_info)

        if (
            not after
        ):  # If there is no space after the word, it means it's a conjunction like ו or ב.
            sub_word_num += 1
        else:
            sub_word_num = 1
            word_line_num += 1

        filtered_data[scroll].append(word_entry)
    return filtered_data


def process_scrolls(specific_scrolls=None):
    filtered_data = defaultdict(list)
    for scroll_node in tqdm(F.otype.s("scroll")):
        if specific_scrolls:
            if not A.sectionStrFromNode(scroll_node) in specific_scrolls:
                continue
        word_line_num = 1
        sub_word_num = 1
        scroll_data = process_word(scroll_node, word_line_num, sub_word_num)
        filtered_data.update(scroll_data)
    return filtered_data
