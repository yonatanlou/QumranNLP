import os

from config import BASE_DIR

from src.baselines.main import make_baselines_results

RERUN = True


def generate_params(n):
    mfw_string = f"MFW={n}"
    NO_PREPROCESSING = []
    WEAK = [mfw_string, "LEMMATIZATION"]
    MEDIUM = [mfw_string, "LEMMATIZATION", "STOPWORDS"]
    STRONG = [mfw_string, "LEMMATIZATION", "STOPWORDS", "LEX"]

    params = [
        ("NO_PREPROCESSING", NO_PREPROCESSING),
        (f"WEAK_{mfw_string}", WEAK),
        (f"MEDIUM_{mfw_string}", MEDIUM),
        (f"STRONG_{mfw_string}", STRONG),
    ]

    return params


# params = generate_params(5)
# params.extend(generate_params(10))
# params.extend(generate_params(25))
# params.extend(generate_params(50))
params = [("LEX", ["LEX"])]
print(params)
from datetime import datetime
from tqdm import tqdm

ALL_EXP_BASE_DIR = f"{BASE_DIR}/experiments/dss/pre_processing"
if RERUN:
    from src.data_generation.dss_data_gen import generate_data

    for name, pre_processing_tasks in tqdm(params):
        name = f"{name}"
        print(f"{datetime.now()} - {pre_processing_tasks=}")

        exp_name = f"{name}"
        exp_dir = f"{ALL_EXP_BASE_DIR}/{exp_name}"

        if not os.path.exists(exp_dir) or len(os.listdir(exp_dir)) == 0:
            os.makedirs(exp_dir, exist_ok=True)
            df_name = f"{exp_dir}/{exp_name}.csv"
            # creating_data
            tmp_df = generate_data(15, 100, pre_processing_tasks, df_name)
            df_filtered_name = f"{exp_dir}/filtered_{exp_name}.csv"
            # making baselines
            make_baselines_results(
                data_path=df_filtered_name,
                processed_vectorizers_path=f"{exp_dir}/{exp_name}.pkl",
                results_dir=exp_dir,
                train_frac=0.7,
                val_frac=0.1,
                tasks=["scroll", "composition"],
            )
