import os
from datetime import datetime
from transformers import BertTokenizer

from config import BASE_DIR
from src.models.BertClassifier import BertClassifier, Dataset, train
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype
from logger import get_logger

MODEL_NAME = "onlplab/alephbert-base"
RUN_NAME = "alephbert-base_sectarian_from_books_to_read_tf_data_sampled"
BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-6
CLASSIFICATION = "sectarian"
LABEL_MAPS = {
    "secterian": {"sectarian_texts": 1, "non_sectarian_texts": 0},
    "bib": {"bib": 1, "nonbib": 0},
}
LABEL_MAP = LABEL_MAPS[CLASSIFICATION]

if __name__ == "__main__":
    model = BertClassifier(MODEL_NAME)
    tokenizer = BertTokenizer.from_pretrained("onlplab/alephbert-base")

    labels_path = "/Users/yonatanlou/dev/QumranNLP/data/hebrew_processed_files/2024-04-09_sectarian_classification/DSS_sectarian_tf_labels.txt"
    texts_path = "/Users/yonatanlou/dev/QumranNLP/data/hebrew_processed_files/2024-04-09_sectarian_classification/DSS_sectarian_tf_text.txt"
    df_labels = pd.read_csv(
        labels_path, sep="\t", header=None, names=["id", "set", "label"]
    )
    with open(texts_path, "r") as file:
        texts = file.readlines()
    df_texts = pd.DataFrame(texts, columns=["text"])
    df_texts["text"] = df_texts["text"].str.strip()

    df_data = pd.concat([df_labels, df_texts], axis=1)
    if not is_numeric_dtype(df_data["label"]):
        df_data["label"] = df_data["label"].map(LABEL_MAP)

    train_val_data = df_data[df_data["set"] == "train"]
    test_data = df_data[df_data["set"] == "test"]
    train_data, val_data = train_test_split(
        train_val_data, test_size=0.2, random_state=42
    )

    train_texts, train_labels = train_data["text"].values, train_data["label"].values
    val_texts, val_labels = val_data["text"].values, val_data["label"].values
    test_texts, test_labels = test_data["text"].values, test_data["label"].values

    # Convert to numpy arrays
    train_data_np, train_label_np = np.array(train_texts), np.array(
        train_labels, dtype=int
    )
    val_data_np, val_label_np = np.array(val_texts), np.array(val_labels, dtype=int)
    test_data_np, test_label_np = np.array(test_texts), np.array(test_labels, dtype=int)
    train_dataset = Dataset(train_data_np, train_label_np, MODEL_NAME)
    val_dataset = Dataset(val_data_np, val_label_np, MODEL_NAME)
    test_dataset = Dataset(test_data_np, test_label_np, MODEL_NAME)

    models_dir = os.path.join(BASE_DIR, "models", "bert_classifier")
    os.makedirs(os.path.join(models_dir, RUN_NAME), exist_ok=True)
    model_name = f"{RUN_NAME}_{datetime.now().strftime('%Y-%m-%d')}"
    logger = get_logger(__name__, f"{models_dir}/{RUN_NAME}/{model_name}.log")
    logger.info(
        f"Train data shape: {train_data_np.shape}, "
        f"Train label shape: {train_label_np.shape}, "
        f"Validation data shape: {val_data_np.shape}, "
        f"Validation label shape: {val_label_np.shape}, "
        f"Test data shape: {test_data_np.shape}, "
        f"Test label shape: {test_label_np.shape}"
    )
    np.save(f"{models_dir}/{RUN_NAME}/test_data.npy", test_data_np)
    np.save(f"{models_dir}/{RUN_NAME}/test_label.npy", test_label_np)

    train(
        model,
        train_dataset,
        val_dataset,
        LR,
        EPOCHS,
        BATCH_SIZE,
        RUN_NAME,
        logger,
        models_dir,
    )
