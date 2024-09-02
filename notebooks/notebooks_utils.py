import re

chars_to_delete = re.compile("[\\\\\^><»≥≤/?Ø\\]\\[«|}{]")


def write_data(data, filename):
    with open(filename, "w") as f:
        for book in data.keys():
            for line in data[book]:
                line = "".join(line[0]) + "\n"

                f.write(line)
