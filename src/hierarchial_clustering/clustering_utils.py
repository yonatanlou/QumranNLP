import utils
from config import BASE_DIR


def generate_books_dict(books_to_run, yaml_book_file):
    book_yml = utils.read_yaml(f"{BASE_DIR}/data/yamls/{yaml_book_file}")
    if any(books_to_run):
        book_dict = {
            k: v for d in book_yml.values() for k, v in d.items() if k in books_to_run
        }
    else:
        book_dict = {k: v for d in book_yml.values() for k, v in d.items()}
    book_to_section = {b: s for s, d in book_yml.items() for b in d}
    return book_dict, book_to_section
