import numpy as np
from transformers import BertTokenizer

from src.hierarchial_clustering.clustering_utils import generate_books_dict
from src.models.BertClassifier import (
    BertClassifier,
    Dataset,
    set_run_name_global,
    train,
)
from src.parsers import parser_data
from src.features.BERT.bert import aleph_bert_preprocessing
from logger import get_logger

RUN_NAME = "bib_nonbib_from_books_to_read"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-6

if __name__ == "__main__":
    model = BertClassifier()
    tokenizer = BertTokenizer.from_pretrained("onlplab/alephbert-base")

    section_data = {"bib": "dss_bib.yaml", "nonbib": "dss_nonbib.yaml"}
    section_type = ["bib", "nonbib"]

    # section_type = ['sectarian_texts', 'non_sectarian_texts']
    train_data, train_label = np.array([]), np.array([], dtype=int)
    val_data, val_label = np.array([]), np.array([], dtype=int)
    test_data, test_label = np.array([]), np.array([], dtype=int)

    size = [0 for i in section_type]
    for i in range(len(section_type)):
        all_data = np.array([])
        all_labels = np.array([], dtype=int)
        section = section_type[i]
        book_dict, book_to_section = generate_books_dict([None], section_data[section])
        data = parser_data.get_dss_data(book_dict, section)
        for book_name, book_data in data.items():
            if len(book_data) < 50:
                print(f"{book_name} have less than 50 samples")
                continue
            book_scores = [section, book_name]
            samples, sample_names = parser_data.get_samples(book_data)
            preprocessed_samples = aleph_bert_preprocessing(samples)
            labels = [i for _ in range(len(samples))]
            all_data = np.concatenate((all_data, preprocessed_samples))
            all_labels = np.concatenate((all_labels, labels))
        size[i] = len(all_data)
        idx = np.arange(len(all_data))
        np.random.shuffle(idx)
        test_size = int(len(all_data) * 0.15)
        test_data, test_label = np.concatenate(
            (test_data, all_data[:test_size])
        ), np.concatenate((test_label, all_labels[:test_size]))
        val_data, val_label = np.concatenate(
            (val_data, all_data[test_size : 2 * test_size])
        ), np.concatenate((val_label, all_labels[test_size : 2 * test_size]))
        train_data, train_label = np.concatenate(
            (train_data, all_data[2 * test_size :])
        ), np.concatenate((train_label, all_labels[2 * test_size :]))

    print(size)
    train_dataset = Dataset(train_data, train_label)
    val_dataset = Dataset(val_data, val_label)
    test_dataset = Dataset(test_data, test_label)
    train(model, train_dataset, val_dataset, LR, EPOCHS, BATCH_SIZE, RUN_NAME)
