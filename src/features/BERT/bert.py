from transformers import BertModel, BertTokenizer
from src.features.BERT.bert_utils import get_preds, get_sentence_vectors
import re

from config import BASE_DIR
from utils import Transcriptor
from logger import get_logger

logger = get_logger(__name__)
chars_to_delete = re.compile("[\\\\\^><»≥≤/?Ø\\]\\[«|}{]")
modes = [
    ["average", 4, "average_word_vectors"],
    ["concat", 4, "average_word_vectors"],
    ["last", 1, "average_word_vectors"],
    ["", "", "CLS embedding"],
]


def aleph_bert_preprocessing(samples):
    # TODO fix parsing issues with ( and ) or unrecognized characters
    transcriptor = Transcriptor(f"{BASE_DIR}/data/yamls/heb_transcript.yaml")
    transcripted_samples = []
    for sample in samples:
        last_entry_word = "X"
        transcripted_sample = []
        word = []
        for entry in sample:
            filtered_entry_transcript = chars_to_delete.sub("", entry["transcript"])
            filtered_entry_transcript = filtered_entry_transcript.replace("\xa0", "")
            # if "(" in filtered_entry_transcript or ")" in filtered_entry_transcript:
            #     old_filtered_entry_transcript = filtered_entry_transcript
            #     filtered_entry_transcript = filtered_entry_transcript.replace(
            #         "(", ""
            #     ).replace(")", "")
            #     logger.info(
            #         f"removed ( and ) from transcript: {old_filtered_entry_transcript} -> {filtered_entry_transcript} ({entry['book_name']}, {entry['chapter_name']})"
            #     )

            if last_entry_word == entry["word_line_num"] or entry["transcript"] == ".":
                word.append(transcriptor.latin_to_heb(filtered_entry_transcript, entry))
            else:
                if len(word):
                    transcripted_sample.append("".join(word))
                word = [transcriptor.latin_to_heb(filtered_entry_transcript, entry)]
            last_entry_word = entry["word_line_num"]
        transcripted_samples.append(transcripted_sample)
    return [" ".join(x) for x in transcripted_samples]


def get_aleph_bert_features(samples, mode_idx):
    preprocessed_samples = aleph_bert_preprocessing(samples)
    alephbert_tokenizer = BertTokenizer.from_pretrained("onlplab/alephbert-base")
    alephbert = BertModel.from_pretrained(
        "onlplab/alephbert-base", output_hidden_states=True
    )
    pretrained_preds = get_preds(
        preprocessed_samples, alephbert_tokenizer, alephbert
    )  # TODO understand
    mode = modes[mode_idx]  # TODO make it configurable (and understand)
    sentence_vectors = get_sentence_vectors(
        pretrained_preds,
        preprocessed_samples,
        wrd_vec_mode=mode[0],
        wrd_vec_top_n_layers=mode[1],
        sentence_emb_mode=mode[2],
        plt_xrange=None,
        plt_yrange=None,
        plt_zrange=None,
        title_prefix="Pretrained model:",
    )
    return sentence_vectors
