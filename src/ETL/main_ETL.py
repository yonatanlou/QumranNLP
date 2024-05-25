import os

import click
from config import BASE_DIR
from logger import get_logger
from src.ETL.ETL_utils import process_scrolls_to_features, filter_df_by_rules
from src.ETL.generate_raw_data import process_scrolls

logger = get_logger(__name__)
WORDS_PER_SAMPLE = 100
SENTENCE_DIVIDER = "׃ "
OUTPUT_FILE = f"{BASE_DIR}/notebooks/data/text_and_starr_features_22_05_2024.csv"


@click.command()
@click.option(
    "--words_per_sample", default=WORDS_PER_SAMPLE, help="Number of words per sample."
)
@click.option("--output_file", default=OUTPUT_FILE, help="Output CSV file path.")
def main(words_per_sample, output_file):
    logger.info("Extracting raw data from text-fabric")
    raw_data = process_scrolls()
    logger.info("Processing scrolls to text and starr features")
    df = process_scrolls_to_features(raw_data, words_per_sample)
    df.to_csv(output_file, index=False)
    df_filtered = filter_df_by_rules(df)
    directory, filename = os.path.split(output_file)
    output_file_filtered = os.path.join(directory, "filtered_" + filename)
    df_filtered.to_csv(output_file_filtered, index=False)
    print("done")


if __name__ == "__main__":
    main()
# python main_ETL.py --words_per_sample 100 --sentence_divider "׃ " --output_file "/path/to/output.csv"
