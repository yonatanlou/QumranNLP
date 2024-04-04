import logging


def get_logger(file_name, log_file_path=None):
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)

    # Create handlers: one for the console, one for the file
    console_handler = logging.StreamHandler()

    # Define a formatter
    formatter = logging.Formatter(
        "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        "%m-%d %H:%M:%S",
    )

    # Add the formatter to the handlers
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)

    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
