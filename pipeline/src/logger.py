import logging

import env


LOGFMT = "%(asctime)s [%(name)s | %(levelname)s]: %(message)s"
DATEFMT = "%Y-%m-%dT%H:%M:%SZ"
logging.basicConfig(format=LOGFMT, datefmt=DATEFMT, level=env.LOGLEVEL)


def get_logger(logger_name: str, log_filepath=env.LOGFILE_PATH) -> logging.Logger:
    """
    Get logger

    Args:
    -----
        logger_name (str):
            Logger name

        log_filepath (str):
            Log filepath

    Returns:
    -----
        logger (logging.Logger):
            Logger
    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(env.LOGLEVEL)

    if log_filepath:
        handler = logging.FileHandler(filename=log_filepath)
        formatter = logging.Formatter(fmt=LOGFMT, datefmt=DATEFMT)
        handler.setFormatter(fmt=formatter)
        logger.addHandler(handler)

    return logger
