import click
from config import BASE_DIR
from logger import get_logger
from src.ETL.ETL_utils import process_scrolls_to_features
from src.ETL.generate_raw_data import process_scrolls

logger = get_logger(__name__)
WORDS_PER_SAMPLE = 100
SENTENCE_DIVIDER = "׃ "
OUTPUT_FILE = f"{BASE_DIR}/notebooks/data/text_and_starr_features_22_05_2024.csv"


@click.command()
@click.option(
    "--words_per_sample", default=WORDS_PER_SAMPLE, help="Number of words per sample."
)
@click.option(
    "--sentence_divider", default=SENTENCE_DIVIDER, help="Divider for sentences."
)
@click.option("--output_file", default=OUTPUT_FILE, help="Output CSV file path.")
def main(words_per_sample, sentence_divider, output_file):
    logger.info("Extracting raw data from text-fabric")
    filtered_data = process_scrolls()
    # import pickle
    # with open(f"{BASE_DIR}/notebooks/data/filtered_data.pkl", "rb") as f:
    #     filtered_data = pickle.load(f)
    logger.info("Processing scrolls to text and starr features")
    df = process_scrolls_to_features(
        filtered_data, words_per_sample, sentence_divider=sentence_divider
    )
    df.to_csv(output_file, index=False)
    print("done")


if __name__ == "__main__":
    main()
# python main_ETL.py --words_per_sample 100 --sentence_divider "׃ " --output_file "/path/to/output.csv"
