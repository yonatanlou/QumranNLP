import os

import click
from config import BASE_DIR
from logger import get_logger
from src.ETL.ETL_utils import process_scrolls_to_features, filter_df_by_rules
from src.ETL.generate_raw_data import process_scrolls

logger = get_logger(__name__)
CHUNK_SIZE = 100
MAX_OVERLAP = 10
OUTPUT_FILE = f"{BASE_DIR}/notebooks/data/text_and_starr_features_{CHUNK_SIZE}_words_nonbib_04_08_2024.csv"


@click.command()
@click.option("--chunk_size", default=CHUNK_SIZE, help="Number of words per sample.")
@click.option(
    "--max_overlap",
    default=MAX_OVERLAP,
    help="Max overlap between chunks (end of i-1 and start of i sample)",
)
@click.option("--output_file", default=OUTPUT_FILE, help="Output CSV file path.")
def main(chunk_size: int, max_overlap: int, output_file: [str, None]):
    logger.info(
        f"Starting ETL process with {chunk_size=}, {max_overlap=}, {output_file=}"
    )
    logger.info("Extracting raw data from text-fabric")
    raw_data = process_scrolls()
    logger.info("Processing scrolls to text and starr features")
    df = process_scrolls_to_features(raw_data, chunk_size, max_overlap=max_overlap)
    df.to_csv(output_file, index=False)
    df_filtered = filter_df_by_rules(df)
    directory, filename = os.path.split(output_file)
    output_file_filtered = os.path.join(directory, "filtered_" + filename)
    if output_file:
        df_filtered.to_csv(output_file_filtered, index=False)
        print(
            f"Saved results to {output_file} (shape:{df.shape}) and {output_file_filtered} (shape:{df_filtered.shape})"
        )
    return df_filtered


def generate_data(chunk_size: int, max_overlap: int, output_file: [str, None]):
    logger.info(
        f"Starting ETL process with {chunk_size=}, {max_overlap=}, {output_file=}"
    )
    logger.info("Extracting raw data from text-fabric")
    raw_data = process_scrolls()
    logger.info("Processing scrolls to text and starr features")
    df = process_scrolls_to_features(raw_data, chunk_size, max_overlap=max_overlap)
    df.to_csv(output_file, index=False)
    df_filtered = filter_df_by_rules(df)
    directory, filename = os.path.split(output_file)
    output_file_filtered = os.path.join(directory, "filtered_" + filename)

    df_filtered.to_csv(output_file_filtered, index=False)
    print(
        f"Saved results to {output_file} (shape:{df.shape}) and {output_file_filtered} (shape:{df_filtered.shape})"
    )
    return df_filtered


# if __name__ == "__main__":
#     main()
# python main_ETL.py --chunk_size 100 --max_overlap 10 --output_file "/path/to/output.csv"
