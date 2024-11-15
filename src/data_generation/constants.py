import string

ALLOWED_CHARS = "אבגדהוזחטיכלמנסעפצקרשתםןףךץ. 1234567890"
MIN_WORDS_PER_SCROLL = 300
MFW_THRESHOLD = 10

NOT_HEB_BOOKS = [
    "1Q20",
    "1Q21",
    "1Q23",
    "1Q24",
    "1Q32",
    "1Q72",
    "2Q24",
    "2Q26",
    "2Q27",
    "2Q28",
    "2Q29",
    "2Q30",
    "2Q31",
    "2Q32",
    "2Q33",
    "2QX1",
    "3Q12",
    "3Q13",
    "3Q14",
    "4Q119",
    "4Q120",
    "4Q121",
    "4Q122",
    "4Q126",
    "4Q127",
    "4Q156",
    "4Q157",
    "4Q196",
    "4Q197",
    "4Q198",
    "4Q199",
    "4Q201a",
    "4Q213",
    "4Q246",
    "4Q539",
    "4Q541",
    "4Q542",
    "4Q555",
    "5Q15",
    "6Q8",
    "6Q14",
    "6Q19",
    "6Q23",
    "6Q25",
    "6Q26",
    "6Q31",
    "7Q1",
    "7Q2",
    "7Q3",
    "7Q4",
    "7Q5",
    "7Q6–18",
    "7Q19",
    "11Q10",
    "11Q18",
    "11Q24",
    "11Q29",
    "11Q31",
    "Mur6",
    "4Q213a",
    "4Q213b",
    "4Q214",
    "4Q214a",
    "4Q214b",
]


manually_remove_scrolls = [
    # aramic:
    "1Q20",
    "1Q21",
    "1Q23",
    "1Q24",
    "1Q32",
    "2Q24",
    "2Q26",
    "4Q318",
    "4Q339",
    "4Q523",
    "11Q18",
    "4Q201",
    "4Q202",
    "4Q203",
    "4Q204",
    "4Q205",
    "4Q206",
    "4Q207",
    "4Q208",
    "4Q209",
    "4Q21",
    "4Q210",
    "4Q211",
    "4Q212",
    "4Q213",
    "4Q213a",
    "4Q213b",
    "4Q214",
    "4Q214a",
    "4Q214b",  # 4Q201 to 4Q214b
    *["4Q" + str(i) for i in range(196, 200)],  # 4Q196 to 4Q199
    *["4Q" + str(i) for i in range(242, 246)],  # 4Q242 to 4Q245
    *["4Q" + str(i) for i in range(529, 570)],  # 4Q529 to 4Q569
    # too short and fragmentary
    *["1Q" + str(i) for i in range(41, 71)],  # 1Q41 to 1Q70
    *["2Q" + str(i) for i in range(22, 33)],  # 2Q22 to 2Q32
    *["4Q" + str(i) for i in range(234, 236)],  # 4Q234, 4Q235
    "4Q238",
    "4Q249",
    "4Q250",
    "4Q313a",
    "4Q313b",
    *[
        "4Q" + str(i) + suffix
        for i in range(281, 283)
        for suffix in [""] + list(string.ascii_lowercase)
    ],  # 4Q281 to 4Q282 with all subsections
    *[
        "4Q" + str(i) + suffix
        for i in range(291, 295)
        for suffix in [""] + list(string.ascii_lowercase)
    ],
    *[
        "4Q" + str(i) + suffix
        for i in range(249, 300)
        for suffix in [""] + list(string.ascii_lowercase)
    ],  # 4Q249 with all subsections
    "4Q307",
    "4Q313",
    "4Q338",
    "4Q340",
    *[
        "4Q" + str(i) + suffix for i in range(360, 361) for suffix in ["", "a"]
    ],  # 4Q360, 4Q360a
    *["4Q" + str(i) for i in range(441, 448)],  # 4Q441 to 4Q447
    *["4Q" + str(i) for i in range(449, 460)],  # 4Q449 to 4Q459
    "4Q464a",
    "4Q464b",
    "4Q465",
    *[
        "4Q" + str(i) + suffix
        for i in range(466, 469)
        for suffix in [""] + list(string.ascii_lowercase)
    ],  # 4Q466 to 4Q468 with all subsections
    "4Q468aa",
    "4Q468bb",
    "4Q468cc",
    "4Q468dd",
    "4Q469",
    "4Q471a",
    "4Q471n",
    "4Q272",
    "4Q473",
    *[
        "4Q" + str(i) + suffix
        for i in range(478, 482)
        for suffix in [""] + list(string.ascii_lowercase)
    ],  # 4Q478 to 4Q481 with all subsections
    *["4Q" + str(i) for i in range(484, 490)],  # 4Q484 to 4Q489
    *["4Q" + str(i) for i in range(498, 501)],  # 4Q498 to 4Q500
    *["4Q" + str(i) for i in range(515, 521)],  # 4Q515 to 4Q520
    *["4Q" + str(i) for i in range(526, 529)],  # 4Q526 to 4Q528
    *["4Q" + str(i) for i in range(570, 588)],  # 4Q570 to 4Q587
    "11Q15",
    "11Q16",
    *["11Q" + str(i) for i in range(22, 28)],  # 11Q22 to 11Q27
    *["3Q" + str(i) for i in range(1, 15) if i != 15],  # Delete all 3Q except 3Q15
    *[
        "4Q" + str(i) for i in range(341, 360)
    ],  # 4Q341 to 4Q359 (probably not from Qumran)
    *[
        "5Q" + str(i) for i in range(1, 26) if i not in [12, 13]
    ],  # Delete all 5Q except 5Q12, 5Q13
    *["6Q" + str(i) for i in range(1, 31) if i != 15],  # Delete all 6Q except 6Q15
    *["8Q" + str(i) for i in range(1, 6)],  # Delete all 8Q
    *["9Q" + str(i) for i in range(1, 10)],  # Delete all 9Q
    *["10Q" + str(i) for i in range(1, 10)],  # Delete all 10Q
    "Pam43113",
    "Pam43124",
    "PAM43660",
    "PAM43661",
    "PAM43663",
    "PAM43664",
    "PAM43665",
    "PAM43666",
    "PAM43667",
    "PAM43668",
    "PAM43669",
    "PAM43670",
    "PAM43671",
    "PAM43672",
    "PAM43673",
    "PAM43674",
    "PAM43675",
    "PAM43676",
    "PAM43677",
    "PAM43678",
    "PAM43679",
    "PAM43680",
    "PAM43682",
    "PAM43683",
    "PAM43684",
    "PAM43685",
    "PAM43686",
    "PAM43688",
    "PAM43689",
    "PAM43690",
    "PAM43691",
    "PAM43692",
    "PAM43693",
    "PAM43694",
    "PAM43695",
    "PAM43696",
    "PAM43697",
    "PAM43698",
    "PAM43699",
    "PAM43700",
    "PAM43701",
    "PAM44102",  # Delete all PAM entries
    "Xq1",
    "Xq2",
    "Xq3",
    "XQ6",
    "XQ7",
    "XQ8",  # Delete XQ, KhQ
]
