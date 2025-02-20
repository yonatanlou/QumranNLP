import time


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
