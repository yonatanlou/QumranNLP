from config import BASE_DIR
from src.parsers.text_reader import read_text
import pandas as pd
from src.parsers.MorphParser import MorphParser
import yaml


text_file_dss_nonbib = f"{BASE_DIR}/data/texts/abegg/dss_nonbib.txt"
yaml_dir = f"{BASE_DIR}/data/yamls"

morph_parser = MorphParser(yaml_dir=yaml_dir)
data, lines = read_text(text_file_dss_nonbib)
res = []
for i in data:
    res.append({"book": i["book_name"], "scroll": i["scroll_name"]})
df = pd.DataFrame(res)
df = df.drop_duplicates()
all_scrolls = df["scroll"].to_list()
book_scroll_dict_ = {"all_scrolls": all_scrolls}
yaml_lines = [f"{book}: {scrolls}" for book, scrolls in book_scroll_dict_.items()]


yaml_content = "\n".join(yaml_lines)
with open(f"{BASE_DIR}/data/yamls/dss_nonbib.yaml", "w") as file:
    file.write(yaml_content)

print("The YAML file has been saved.")
