import pickle


from config import BASE_DIR
from src.baselines.embeddings import VectorizerProcessor, get_vectorizer_types
from src.gnn.hyperparameter_gnn_utils import run_gnn_exp
from src.gnn.utils import create_param_dict
from itertools import product, combinations
import os.path

PROCESSED_VECTORIZERS_PATH = f"{BASE_DIR}/src/data/processed_vectorizers.pkl"
EXP_NAME = "gcn_bert_ensemble"
NUM_WORD_PER_CHUNK = 100
NUM_COMBINED_GRAPHS = 3

with open(f"{BASE_DIR}/src/data/datasets.pkl", "rb") as f:
    datasets = pickle.load(f)

params = {
    "epochs": [500],
    "hidden_dims": [300],
    "distances": ["cosine"],
    "learning_rates": [0.001],
    "thresholds": [0.98, 0.99],
    "bert_models": [
        "dicta-il/BEREL",
        # "onlplab/alephbert-base",
        "yonatanlou/BEREL-finetuned-DSS-maskedLM",
        # "yonatanlou/BEREL-finetuned-DSS-composition-classification",
    ],
    "adj_types": {
        "tfidf": {"max_features": 7500},
        "trigram": {"analyzer": "char", "ngram_range": (3, 3)},
        "BOW-n_gram": {"analyzer": "word", "ngram_range": (1, 1)},
        "bert-berel": {"type": "dicta-il/BEREL"},
        "bert-alephbert": {"type": "onlplab/alephbert-base"},
        "bert-finetune-lm": {"type": "yonatanlou/BEREL-finetuned-DSS-maskedLM"},
    },
}


all_param_dicts = []

meta_param_combinations = product(
    params["epochs"],
    params["thresholds"],
    params["distances"],
    params["hidden_dims"],
    params["learning_rates"],
    params["bert_models"],
)

for epoch, threshold, distance, hidden_dim, lr, bert_model in meta_param_combinations:
    meta_params = {
        "epochs": epoch,
        "hidden_dim": hidden_dim,
        "distance": distance,
        "learning_rate": lr,
        "threshold": threshold,
        "bert_model": bert_model,
    }
    for n in range(1, NUM_COMBINED_GRAPHS + 1):
        adj_combinations = combinations(params["adj_types"].items(), n)
        all_param_dicts.extend(create_param_dict(n, adj_combinations, meta_params))


for dataset_name, dataset in datasets.items():
    if dataset_name == "dataset_scroll":
        continue
    print(f"starting with {dataset_name}")
    exp_dir_path = f"{BASE_DIR}/reports/gnn/{EXP_NAME}"
    if not os.path.exists(exp_dir_path):
        os.makedirs(exp_dir_path)
    file_name = f"{exp_dir_path}/{EXP_NAME}_{dataset.label}_{NUM_WORD_PER_CHUNK}_words_{NUM_COMBINED_GRAPHS}_adj_types.csv"
    if os.path.isfile(file_name):
        continue
    df = dataset.df
    vectorizer_types = get_vectorizer_types()

    processor = VectorizerProcessor(df, PROCESSED_VECTORIZERS_PATH, vectorizer_types)
    processed_vectorizers = processor.load_or_generate_embeddings()
    df = df.reset_index()

    run_gnn_exp(
        all_param_dicts, df, processed_vectorizers, file_name, dataset, verbose=False
    )
