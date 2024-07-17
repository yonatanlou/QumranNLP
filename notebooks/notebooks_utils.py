import random

import numpy as np
import pandas as pd
import re

import torch

from notebooks.constants import MIN_WORDS_PER_BOOK

chars_to_delete = re.compile("[\\\\\^><»≥≤/?Ø\\]\\[«|}{]")


def write_data(data, filename):
    with open(filename, "w") as f:
        for book in data.keys():
            for line in data[book]:
                line = "".join(line[0]) + "\n"

                f.write(line)


def set_seed_globally(seed=42):
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
