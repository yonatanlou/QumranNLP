import logging


def get_logger(file_name):
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)

    # define handler and formatter
    handler = logging.StreamHandler()
    # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    formatter = logging.Formatter(
        "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
        "%m-%d %H:%M:%S",
    )

    # add formatter to handler
    handler.setFormatter(formatter)

    # add handler to logger
    logger.addHandler(handler)
    return logger
