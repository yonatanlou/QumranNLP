from config import BASE_DIR
from src.parsers.text_reader import read_text
import pandas as pd
from src.parsers.MorphParser import MorphParser
import yaml

text_file_dss_bib = f"{BASE_DIR}/data/texts/abegg/dss_bib.txt"
text_file_dss_nonbib = f"{BASE_DIR}/data/texts/abegg/dss_nonbib.txt"
yaml_dir = f"{BASE_DIR}/data/yamls"

morph_parser = MorphParser(yaml_dir=yaml_dir)
data, lines = read_text(text_file_dss_bib)
res = []
for i in data:
    res.append({"book": i["book_name"], "scroll": i["scroll_name"]})
df = pd.DataFrame(res)
df = df.drop_duplicates()
book_scroll_dict = df.groupby("book")["scroll"].apply(list).to_dict()
book_scroll_dict_ = {}
for k, v in book_scroll_dict.items():
    if (k == v[0]) and (len(v) == 1):
        continue
    else:
        book_scroll_dict_[k] = v

yaml_lines = [f"{book}: {scrolls}" for book, scrolls in book_scroll_dict_.items()]


yaml_content = "\n".join(yaml_lines)
with open(f"{BASE_DIR}/data/yamls/dss_bib.yaml", "w") as file:
    file.write(yaml_content)

print("The YAML file has been saved.")
