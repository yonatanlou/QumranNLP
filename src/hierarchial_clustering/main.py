from config import BASE_DIR
from src.hierarchial_clustering.constants import TRIGRAM_FEATURE_LENGTH
from src.hierarchial_clustering.hierarchical_clustering import main, get_bar_graph

RESULTS_PATH = f"{BASE_DIR}/results/Clusters_reconstruction"
BOOKS_TO_RUN_ON = [
    "Musar Lamevin",
    "Berakhot",
    "4QM",
    "Catena_Florilegium",
    "4QD",
    "Hodayot",
    "Pesharim",
    "1QH",
    "1QS",
    "Songs_of_Maskil",
    "non_biblical_psalms",
    "Mysteries",
    "4QS",
    "4QH",
    "1QSa",
    "CD",
    "1QM",
    "Collections_of_psalms",
    "Book_of_Jubilees",
]

main(
    "all_sectarian_texts.yaml",
    BOOKS_TO_RUN_ON,
    "nonbib",
    RESULTS_PATH,
    TRIGRAM_FEATURE_LENGTH,
)


get_bar_graph(
    ["bert", "trigram", "starr", "bert_matmul_trigram", "bert_concat_trigram"],
    RESULTS_PATH,
)
