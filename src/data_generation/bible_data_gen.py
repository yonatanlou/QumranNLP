import os

import click
import pandas as pd
from tqdm import tqdm
from tf.app import use
from typing import Dict, List, Tuple
from dataclasses import dataclass
import re
from datetime import datetime
from config import BASE_DIR

A = use("etcbc/bhsa", hoist=globals())
ALLOWED_CHARS = "-אבגדהוזחטיכלמנסעפצקרשתםןףךץ. 1234567890" + "\u05BE"  # hebrew makaf


@dataclass
class WordInfo:
    text: str
    chapter: int
    verse: int
    morphology: dict
    position: int  # Position in the book
    raw_text: str  # Store the original text before cleaning


@dataclass
class Chunk:
    text: str
    start_path: Tuple[int, int]  # (chapter, verse) of first word
    end_path: Tuple[int, int]  # (chapter, verse) of last word
    start_position: int
    end_position: int
    words: List[WordInfo]
    book: str


class HebrewBibleProcessor:
    def __init__(self, limit=None):
        # Initialize text-fabric with BHSA
        self.books_data = {}
        self.process_all_books(limit)

    @staticmethod
    def remove_not_heb_chars(word: str) -> str:
        return "".join(char for char in word if char in ALLOWED_CHARS)

    @staticmethod
    def remove_hebrew_punctuation(text: str) -> str:
        """Remove Hebrew niqqud while preserving makaf (hyphen)."""

        hebrew_punctuation = r"[\u0591-\u05BD\u05BF-\u05C7]+"
        heb_no_nikud = re.sub(hebrew_punctuation, "", text)
        heb_no_nikud = heb_no_nikud.replace("\uFB2A", "\u05E9").replace(
            "\uFB2B", "\u05E9"
        )  # שׁ to ש
        return heb_no_nikud

    def clean_text(self, text: str) -> str:
        """Apply all text cleaning rules."""
        text = text.replace("׃", "").replace("׳", "")
        text = self.remove_hebrew_punctuation(text)
        text = self.remove_not_heb_chars(text)
        text = re.sub(" +", " ", text)  # Remove extra spaces
        text = text.replace("\xa0", "")  # Remove non-breaking spaces

        return text

    def get_word_morphology(self, word_node) -> dict:
        """Extract morphological features for a word"""
        return {
            "lemma": T.text(word_node),
            "pos": F.sp.v(word_node),  # part of speech
            "stem": F.vs.v(word_node),  # verbal stem
        }

    def process_all_books(self, limit=None):
        """Process all books and store word-level information"""
        if limit == None:
            limit = len(F.otype.s("book"))
        for book in tqdm(F.otype.s("book")[:limit], desc="Processing books"):
            book_name = T.sectionFromNode(book)[0]
            self.books_data[book_name] = self.process_book(book)

    def process_book(self, book_node) -> Dict[int, WordInfo]:
        """Process a single book and return word-level information"""
        word_dict = {}
        position = 0

        # Get all verses in this book
        for verse in L.d(book_node, otype="verse"):
            # Get chapter and verse numbers
            chapter, verse_num = T.sectionFromNode(verse)[1:3]

            # Process all words in this verse
            for word in L.d(verse, otype="word"):
                raw_text = T.text(word)
                cleaned_text = self.clean_text(raw_text)

                # Only include words that aren't empty after cleaning
                if cleaned_text:
                    word_dict[position] = WordInfo(
                        text=cleaned_text,
                        chapter=chapter,
                        verse=verse_num,
                        morphology=self.get_word_morphology(word),
                        position=position,
                        raw_text=raw_text,
                    )
                    position += 1

        return word_dict

    def get_chunks(
        self, book_name: str, chunk_size: int = 100, overlap: int = 10
    ) -> List[Chunk]:
        if book_name not in self.books_data:
            raise ValueError(f"Book {book_name} not found")

        book_data = self.books_data[book_name]
        chunks = []
        total_words = len(book_data)

        step_size = chunk_size - overlap
        for start_pos in range(0, total_words, step_size):
            end_pos = min(start_pos + chunk_size, total_words)
            chunk_words = [book_data[i] for i in range(start_pos, end_pos)]

            chunk = Chunk(
                text="".join(word.text for word in chunk_words),
                start_path=(chunk_words[0].chapter, chunk_words[0].verse),
                end_path=(chunk_words[-1].chapter, chunk_words[-1].verse),
                start_position=start_pos,
                end_position=end_pos - 1,
                words=chunk_words,
                book=book_name,
            )
            chunks.append(chunk)

        return chunks

    def get_book_names(self) -> List[str]:
        """Get list of all available book names"""
        return sorted(self.books_data.keys())

    def get_word_info(self, book_name: str, position: int) -> WordInfo:
        """Get information about a specific word in a book"""
        if book_name not in self.books_data:
            raise ValueError(f"Book {book_name} not found")
        if position not in self.books_data[book_name]:
            raise ValueError(f"Position {position} not found in {book_name}")
        return self.books_data[book_name][position]

    def get_all_chunks_dataframe(
        self, chunk_size: int = 100, overlap: int = 10
    ) -> pd.DataFrame:
        """Create a DataFrame containing all chunks from all books."""
        all_chunks = []

        for book_name in self.get_book_names():
            book_chunks = self.get_chunks(book_name, chunk_size, overlap)
            all_chunks.extend(book_chunks)

        data = []
        for chunk in all_chunks:
            # Format path as "BookName Chapter:Verse-Chapter:Verse"
            chunk_path = f"{chunk.start_path[0]}:{chunk.start_path[1]}-{chunk.end_path[0]}:{chunk.end_path[1]}"

            data.append(
                {"book": chunk.book, "sentence_path": chunk_path, "text": chunk.text}
            )

        return pd.DataFrame(data)


def generate_data_bible(chunk_size, max_overlap, output_file):
    limit = 6
    if limit != None:
        print(
            "Note! you will produce only 6 books of the Bible. (set limit to None for producing all books)"
        )
    processor = HebrewBibleProcessor()
    df = processor.get_all_chunks_dataframe(chunk_size=chunk_size, overlap=max_overlap)

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")


CHUNK_SIZE = 100
MAX_OVERLAP = 10
DATE = datetime.now().strftime("%Y_%d_%m")
OUTPUT_FILE = (
    f"{BASE_DIR}/data/processed_data/bible/df_{CHUNK_SIZE=}_{MAX_OVERLAP=}_{DATE}.csv"
)


@click.command()
@click.option("--chunk_size", default=CHUNK_SIZE, help="Number of words per sample.")
@click.option(
    "--max_overlap",
    default=MAX_OVERLAP,
    help="Max overlap between chunks (end of i-1 and start of i sample)",
)
@click.option("--output_file", default=OUTPUT_FILE, help="Full path to output file")
def bible_data_gen_main(chunk_size: int, max_overlap: int, output_file: [str, None]):
    return generate_data_bible(chunk_size, max_overlap, output_file)


if __name__ == "__main__":
    bible_data_gen_main()
