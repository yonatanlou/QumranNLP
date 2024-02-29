
import collections
from MorphParser import FieldNames
from logger import get_logger

logger = get_logger(__name__)
line_fields_names = FieldNames()


def verseNum(text):
    if text.isdigit():
        return (text, None)
    return (text[0:-1], text[-1])


def read_text(text_file):
    XC = "\u001b"
    iBIBINFO = "bibinfo"
    iSCROLLINFO = "scrollinfo"
    iSCROLLNAME = "scrollname"
    iSCROLLREF = "scrollref"
    iTRANS = "trans"
    iANALYSIS = "analysis"
    iNUM = "num"

    source_type = text_file.split('_')[-1].split('.')[0]
    split_char = {'bib': '\t',
                  'nonbib':' '}[source_type]
    cols_names = {
        'bib': (iBIBINFO, iSCROLLINFO, iTRANS, iANALYSIS, iNUM),
        'nonbib': (iSCROLLNAME, iSCROLLREF, iTRANS, iANALYSIS,)}[source_type]
    n_cols = len(cols_names)

    # scrollDecl = read_yaml(path.join(yaml_dir, 'scroll.yaml'))
    # fixesDecl = read_yaml(path.join(yaml_dir, 'fixes.yaml'))
    # lineFixes = fixesDecl["lineFixes"]
    # fieldFixes = fixesDecl["fieldFixes"]
    # fixL = "FIX (LINE)"
    # fixF = "FIX (FIELD)"
    # lines = collections.defaultdict(set)

    prev_frag_line_num = None
    prev_word_num = None
    frag_line_num = None
    subNum = None
    interlinear = None
    script = None
    line_num = 0
    parsed_data = []
    lines = []
    line_counter = 0
    with open(text_file) as f:
        for line in f:
            line_counter += 1
            if line_counter % 10_000 == 0:
                logger.info(f"processed {line_counter} lines")
            line_num += 1
            # "check for another language (like greek or paleo hebrew'
            if XC in line:
                xLine = line
                if "(a)" in xLine:
                    interlinear = 1
                elif "(b)" in xLine:
                    interlinear = 2
                elif xLine.startswith(f"{XC}r"):
                    interlinear = ""

                if "(fl)" in xLine:
                    script = 'paleohebrew'
                elif "(f0)" in xLine:
                    script = 'greekcapital'
                elif "(fy)" in xLine:
                    script = ""

                continue
            line = line.rstrip("\n")
            if source_type == 'bib':
                pass

            elif source_type == 'nonbib':
                if line.startswith(">"):
                    line = line[1:]
                    fields = line.split(split_char)
                    scroll = fields[0]
                    (fragment, frag_line_num) = fields[1].split(":", 1)
                    if frag_line_num != prev_frag_line_num:
                        interlinear = ""
                    prev_frag_line_num = frag_line_num
                    continue
            else:
                assert 0, '{} is not a valid source type'.format(source_type)
            fields = line.split(split_char)
            n_fields = len(fields)
            if n_fields > n_cols:
                # diag("FIELDS", f"too many: {nFields}", -1)
                # print('to many fields')
                continue
            elif n_fields < n_cols:
                fields += [""] * (n_cols - n_fields)

            line_data = collections.defaultdict(
                    lambda: "", ((f, c) for (f, c) in zip(cols_names, fields)),
                )

            parsed_word = collections.defaultdict(lambda: "")
            parsed_word[line_fields_names.source_line_num] = line_num
            trans = line_data[iTRANS]
            if source_type == 'bib':
                (scroll, rest) = line_data[iSCROLLINFO].split(" ")
                (fragment, fragment_line_num) = rest.split(":")
                parsed_word[line_fields_names.frag_label] = fragment
                parsed_word[line_fields_names.frag_line_num] = fragment_line_num
                (book, rest) = line_data[iBIBINFO].split(" ", 1)
                (chapter, verse) = rest.split(":", 1)
                parsed_word[line_fields_names.book_name] = book
                parsed_word[line_fields_names.chapter_name] = chapter
                (verse, halfVerse) = verseNum(verse)
                parsed_word[line_fields_names.verse] = verse
                parsed_word[line_fields_names.hverse] = halfVerse
                word = line_data[iNUM]
                parsed_word[line_fields_names.word_line_num] = word
                if "." in word:
                    parsed_word[line_fields_names.word_prefix] = True
            else:
                # some processing of reconstructions
                if trans.startswith("]") and trans.endswith("["):
                    text = trans[1:-1]
                    if text.isdigit():
                        subNum = text[::-1]
                        continue

                (fragment, rest) = line_data[iSCROLLREF].split(":", 1)
                (frag_line_num, all_word) = rest.split(",", 1)
                if frag_line_num != prev_frag_line_num:
                    interlinear = ""
                parsed_word[line_fields_names.frag_label] = fragment
                parsed_word[line_fields_names.frag_line_num] = frag_line_num
                if line == "0":
                    if subNum:
                        parsed_word[line_fields_names.sub_num] = subNum
                (word_num, sub_word_num) = all_word.split(".", 1)
                parsed_word[line_fields_names.word_line_num] = word_num
                parsed_word[line_fields_names.sub_word_num] = sub_word_num
                if word_num == prev_word_num:
                    parsed_data[-1][line_fields_names.word_prefix] = True
                prev_word_num = word_num

            parsed_word[line_fields_names.scroll_name] = scroll

            lines.append((scroll, fragment, line))

            if interlinear:
                parsed_word[line_fields_names.interlinear] = interlinear
            if script:
                parsed_word[line_fields_names.script_type] = script

            analysis = line_data[iANALYSIS] or ""
            (lang, lex, morph) = ("", "", "")
            if "%" in analysis:
                lang = 'aramiac'
                (lex, morph) = analysis.split("%", 1)
            elif "@" in analysis:
                (lex, morph) = analysis.split("@", 1)
            else:
                lex = analysis
            parsed_word[line_fields_names.transcript] = trans
            parsed_word[line_fields_names.lang] = lang
            parsed_word[line_fields_names.lex] = lex
            parsed_word[line_fields_names.morph] = morph

            prev_frag_line_num = frag_line_num
            parsed_data.append(parsed_word)
        return parsed_data, lines