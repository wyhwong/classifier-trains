import logging

import utils.constants
import utils.env

logging.basicConfig(
    format=utils.constants.LOGFMT,
    datefmt=utils.constants.DATEFMT,
    level=utils.env.LOGLEVEL,
)


def get_logger(
    logger_name: str, log_filepath=f"{utils.env.OUTPUT_DIR}/output.log"
) -> logging.Logger:
    """
    Get the logger with the given logger_name and log_filepath.

    Parameters
    ----------
    logger_name: str, required, the name of the logger.
    log_filepath: str, optional, the filepath of the log file.

    Returns
    -------
    logger: logging.Logger, the logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(utils.env.LOGLEVEL)
    if log_filepath:
        handler = logging.FileHandler(filename=log_filepath)
        formatter = logging.Formatter(
            fmt=utils.constants.LOGFMT, datefmt=utils.constants.DATEFMT
        )
        handler.setFormatter(fmt=formatter)
        logger.addHandler(handler)
    return logger
