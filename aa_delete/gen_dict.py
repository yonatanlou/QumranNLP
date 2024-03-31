import yaml

heb_lat = (
    ("א", "a"),
    ("ב", "b"),
    ("ג", "g"),
    ("ד", "d"),
    ("ה", "h"),
    ("ו", "w"),
    ("ז", "z"),
    ("ח", "j"),
    ("ט", "f"),
    ("י", "y"),
    ("כ", "k"),
    ("ך", "K"),
    ("ל", "l"),
    ("מ", "m"),
    ("ם", "M"),
    ("נ", "n"),
    ("ן", "N"),
    ("ס", "s"),
    ("פ", "p"),
    ("ף", "P"),
    ("ע", "o"),
    ("צ", "x"),
    ("ץ", "X"),
    ("ק", "q"),
    ("ר", "r"),
    ("שׂ", "c"),  # FB2B sin
    ("שׁ", "v"),  # FB2A shin
    ("ש" "C"),  # 05E9 dotless shin
    ("ת", "t"),
    # Vowels
    ("\u05b0", "V"),  # sheva
    ("\u05b0", "√"),  # sheva
    ("\u05b0", "J"),  # sheva
    ("\u05b0", "◊"),  # sheva
    ("\u05b1", "T"),  # hataf segol
    ("\u05b2", "S"),  # hataf patah
    ("\u05b3", "F"),  # hataf qamats
    ("\u05b3", "ƒ"),  # hataf qamats
    ("\u05b3", "Ï"),  # hataf qamats
    ("\u05b4", "I"),  # hiriq
    ("\u05b4", "ˆ"),  # hiriq
    ("\u05b4", "î"),  # hiriq
    ("\u05b4", "Ê"),  # hiriq
    ("\u05b5", "E"),  # tsere
    ("\u05b5", "é"),  # tsere
    ("\u05b5", "´"),  # tsere
    ("\u05b6", "R"),  # segol
    ("\u05b6", "®"),  # segol
    ("\u05b6", "‰"),  # segol
    ("\u05b7", "A"),  # patah
    ("\u05b7", "Å"),  # patah
    ("\u05b7", "å"),  # patah
    ("\u05b8", "D"),  # qamats
    ("\u05b8", "∂"),  # qamats
    ("\u05b8", "Î"),  # qamats
    ("\u05b9", "O"),  # holam
    ("\u05b9", "ø"),  # holam
    ("\u05bb", "U"),  # qubbuts
    ("\u05bb", "ü"),  # qubbuts
    ("\u05bb", "¨"),  # qubbuts
    # points
    ("\u05bc", ";"),  # dagesh
    ("\u05bc", "…"),  # dagesh
    ("\u05bc", "Ú"),  # dagesh
    ("\u05bc", "¥"),  # dagesh
    ("\u05bc", "Ω"),  # dagesh
)

heb_to_lat = dict(heb_lat)
lat_to_heb = dict((x[1], x[0]) for x in heb_lat)
to_json_dict = {"heb_to_latin": heb_to_lat, "latin_to_heb": lat_to_heb}

with open("heb_transcript", "w") as fp:
    yaml.dump(to_json_dict, fp, indent=4)
