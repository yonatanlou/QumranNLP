import os
from datetime import datetime
import click
from config import BASE_DIR
from logger import get_logger
from src.ETL.ETL_utils import (
    process_scrolls_to_features,
    filter_df_by_rules,
    add_labels,
)
from src.ETL.generate_raw_data import process_scrolls
import numpy as np

logger = get_logger(__name__)
CHUNK_SIZE = 100
MAX_OVERLAP = 15
# PRE_PROCESSING_TASKS = ["MFW=5", "LEMMATIZATION", "STOPWORDS", "LEX"]
PRE_PROCESSING_TASKS = []
DATE = datetime.now().strftime("%Y_%d_%m")
OUTPUT_FILE = f"{BASE_DIR}/experiments/pre_processing/df_{CHUNK_SIZE=}_{MAX_OVERLAP=}_{PRE_PROCESSING_TASKS=}_{DATE}.csv"


def generate_data(
    max_overlap: int,
    chunk_size: int,
    pre_processing_tasks: list,
    output_file: [str, None],
):
    logger.info(
        f"Starting ETL process with {chunk_size=}, {max_overlap=}, {pre_processing_tasks=},{output_file=}"
    )
    logger.info("Extracting raw data from text-fabric")
    raw_data = process_scrolls()
    logger.info("Processing scrolls to text and starr features")
    df = process_scrolls_to_features(
        chunk_size, raw_data, pre_processing_tasks, max_overlap=max_overlap
    )
    if "LEX" in pre_processing_tasks:
        df["text_lex"] = df["text_lex"].replace("", np.nan)
        df["text_lex"].fillna(df["text"], inplace=True)
        df["text"] = df["text_lex"]
        df["text"].replace("", "ו")
    df = add_labels(df)
    df.to_csv(output_file, index=False)
    df_filtered = filter_df_by_rules(df)
    directory, filename = os.path.split(output_file)
    output_file_filtered = os.path.join(directory, "filtered_" + filename)

    df_filtered.to_csv(output_file_filtered, index=False)
    logger.info(
        f"Saved results to {output_file} (shape:{df.shape}) and {output_file_filtered} (shape:{df_filtered.shape})"
    )
    return df_filtered


@click.command()
@click.option("--chunk_size", default=CHUNK_SIZE, help="Number of words per sample.")
@click.option(
    "--max_overlap",
    default=MAX_OVERLAP,
    help="Max overlap between chunks (end of i-1 and start of i sample)",
)
@click.option(
    "--pre_processing_tasks",
    default=PRE_PROCESSING_TASKS,
    help="one or more of: MFW, LEMMATIZATION, STOPWORDS, LEX",
)
@click.option("--output_file", default=OUTPUT_FILE, help="Full path to output file")
def main(
    chunk_size: int, max_overlap: int, pre_processing_tasks, output_file: [str, None]
):
    pre_processing_tasks = eval(pre_processing_tasks)
    return generate_data(max_overlap, chunk_size, pre_processing_tasks, output_file)


if __name__ == "__main__":
    main()
# python main_ETL.py --chunk_size 100 --max_overlap 10 --output_file "/path/to/output.csv"
