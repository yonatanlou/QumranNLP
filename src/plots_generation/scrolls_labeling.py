# Define your labels with their corresponding ranges
labels_1QS = {
    "1QS=1:1 – 3:12": (1, 1, 3, 12),
    "1QS=3:13 – 4:26": (3, 13, 4, 26),
    "1QS=5:1 – 6:23": (5, 1, 6, 23),
    "1QS=6:24 – 7:25": (6, 24, 7, 25),
    "1QS=8:1 – 8:19": (8, 1, 8, 19),
    "1QS=8:20 – 9:11": (8, 20, 9, 11),
    "1QS=9:12 – 9:26": (9, 12, 9, 26),
    "1QS=10:1 – 11:22": (10, 1, 11, 22),
    "1QSa=all": (1000, 1000, 1000, 1000),
}
labels_hodayot = {
    "1QHa=Community hymns": (3, 1, 8, 28),  # 3:1–8:28
    "1QHa=Transition material": (9, 1, 9, 40),  
    "1QHa=Teacher hymns": (10, 1, 16, 40),  
    "1QHa=Transition material_2": (17, 1, 17, 36),  
    "1QHa=Community hymns_2": (17, 38, 23, 16),  
    "1QHa=Community hymns_3": (24, 1, 24, 1000),  
    "1QHa=Community hymns_4": (27, 1, 27, 1000),  
    "1QHa=allTheRest": (23, 16, 1, 1),
}
# Community Hynms: 3:1–8:28 + 17:38–23:16 + 24:1–24:1000 + 27:1–27:1000
# Transition Material: 9:1–9:40 + 17:1–17:36
# Teacher Hymns: 10:1–16:40
labels_1QM = {
    "1QM=1:1 – 1:18": (1, 1, 1, 18),
    "1QM=allTheRest": (2, 1, 10000, 1000),
}


def parse_range(sentence_path):
    start, end = sentence_path.split("-")
    scroll = start.split(":")[0]

    start = start.split(":")[1:]
    start_chapter, start_verse = map(int, start)

    end_chapter, end_verse = map(int, end.split(":"))
    return scroll, start_chapter, start_verse, end_chapter, end_verse


def calculate_overlap(
    start_chapter,
    start_verse,
    end_chapter,
    end_verse,
    l_start_chapter,
    l_start_verse,
    l_end_chapter,
    l_end_verse,
):
    # Convert the start and end points to a comparable single number (e.g., verse count)
    start = start_chapter * 1000 + start_verse
    end = end_chapter * 1000 + end_verse
    l_start = l_start_chapter * 1000 + l_start_verse
    l_end = l_end_chapter * 1000 + l_end_verse

    # Calculate overlap
    overlap_start = max(start, l_start)
    overlap_end = min(end, l_end)
    overlap = max(0, overlap_end - overlap_start)
    return overlap


def match_label(sentence_path, labels):
    scroll, start_chapter, start_verse, end_chapter, end_verse = parse_range(
        sentence_path
    )
    if scroll == "1QSa":
        return "1QSa"

    max_overlap = 0
    best_label = None

    for label, (
        l_start_chapter,
        l_start_verse,
        l_end_chapter,
        l_end_verse,
    ) in labels.items():
        if label.startswith(scroll):  # Ensure we are matching the correct scroll
            overlap = calculate_overlap(
                start_chapter,
                start_verse,
                end_chapter,
                end_verse,
                l_start_chapter,
                l_start_verse,
                l_end_chapter,
                l_end_verse,
            )
            if overlap > max_overlap:
                max_overlap = overlap
                best_label = label
    if not best_label:
        best_label = [lab for lab in labels.keys() if "allTheRest" in lab][0]
    return best_label


def label_sentence_path(df_labeled_for_clustering, labels, verbose=True):
    sentence_paths = df_labeled_for_clustering["sentence_path"].to_list()
    results = []
    for sentence_path in sentence_paths:
        try:
            label = match_label(sentence_path, labels)
        except Exception:
            label = [lab for lab in labels.keys() if "allTheRest" in lab][0]
        results.append((sentence_path, label))
        if verbose:
            print(f"Sentence Path: {sentence_path} -> Label: {label}")

    df_labeled_for_clustering = df_labeled_for_clustering.copy()
    df_labeled_for_clustering.loc[:, "label"] = [i[1] for i in results]
    df_labeled_for_clustering = df_labeled_for_clustering[
        df_labeled_for_clustering["label"] != "1QHa=allTheRest"
    ]  # not relevant anymore
    df_labeled_for_clustering["label"] = df_labeled_for_clustering["label"].replace({
        "1QHa=Community hymns_2": "1QHa=Community hymns",
        "1QHa=Community hymns_3": "1QHa=Community hymns",
        "1QHa=Community hymns_4": "1QHa=Community hymns",
        "1QHa=Transition material_2": "1QHa=Transition material"
    })



    return df_labeled_for_clustering


### example use
# curr_scroll = ["1QHa"]
# df_labeled_for_clustering = df[df["book"].isin(curr_scroll)]
# df_labeled_for_clustering = label_sentence_path(df_labeled_for_clustering, labels_hodayot,verbose=True)
# df_labeled_for_clustering
