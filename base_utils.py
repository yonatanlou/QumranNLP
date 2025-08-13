import time
import yaml
from collections import defaultdict
def read_yaml(fileName):
    with open(fileName) as y:
        y = yaml.load(y, Loader=yaml.FullLoader)
    return y

def filter_data_by_field(field, books_list, unfiltered_data):
    c_to_book = {v: k for k, l in books_list.items() for v in l}
    # c_to_book = {b: b for b in books_list}
    filtered_data = defaultdict(list)
    for entry in unfiltered_data:
        if entry[field] in c_to_book.keys():
            filtered_data[c_to_book[entry[field]]].append(entry)
    return filtered_data

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture the start time
        result = func(*args, **kwargs)  # Call the decorated function
        end_time = time.time()  # Capture the end time
        print(
            f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds"
        )
        return result  # Return the result of the function call

    return wrapper
