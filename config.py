import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DSS_DATA_PATH = f"{BASE_DIR}/data/processed_data/dss"
DSS_DATA_CSV = f"{DSS_DATA_PATH}/filtered_df_CHUNK_SIZE=100_MAX_OVERLAP=15_PRE_PROCESSING_TASKS=[]_2024_02_09.csv"
DSS_PROCESSED_VECTORIZERS_PATH = f"{DSS_DATA_PATH}/processed_vectorizers.pkl"  # The processed embeddings of DSS_DATA_CSV will be stored here

BIBLE_DATA_PATH = f"{BASE_DIR}/data/processed_data/dss"
BIBLE_DATA_CSV = f"{BIBLE_DATA_PATH}/df_CHUNK_SIZE=100_MAX_OVERLAP=10_2024_15_11.csv"
BIBLE_PROCESSED_VECTORIZERS_PATH = f"{BIBLE_DATA_PATH}/processed_vectorizers.pkl"  # The processed embeddings of DSS_DATA_CSV will be stored here


def get_paths_by_domain(domain):
    if domain == "dss":
        return {
            "data_path": DSS_DATA_PATH,
            "data_csv_path": DSS_DATA_CSV,
            "processed_vectorizers_path": DSS_PROCESSED_VECTORIZERS_PATH,
        }
    if domain == "bible":
        return {
            "data_path": DSS_DATA_PATH,
            "data_csv_path": DSS_DATA_CSV,
            "processed_vectorizers_path": DSS_PROCESSED_VECTORIZERS_PATH,
        }
    else:
        raise ValueError(f"Invalid domain: {domain} (only dss implemented)")
