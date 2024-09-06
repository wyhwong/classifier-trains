import logging
import logging.handlers
import os
import sys
from typing import Optional

from classifier_trains.utils import env


def get_logger(
    logger_name: str,
    streaming_log_level: int = env.STREAMING_LOG_LEVEL,
    file_log_level: int = env.FILE_LOG_LEVEL,
    log_filepath: Optional[str] = env.LOG_FILEPATH,
) -> logging.Logger:
    """Function to create a logger object with the specified name and log levels.

    Args:
        logger_name (str): Name of the logger object.
        streaming_log_level (int, optional): Log level for console logging.
            Defaults to env.STREAMING_LOG_LEVEL.
        file_log_level (int, optional): Log level for file logging.
            Defaults to env.FILE_LOG_LEVEL.
        log_filepath (Optional[str], optional): Path to the log file.
            Defaults to env.LOG_FILEPATH.

    Returns:
        logging.Logger: Logger object with the specified name and log levels.

    Raises:
        FileNotFoundError: If the directory of the log file does not exist.

    Example:
        >>> logger = get_logger("my_logger")
        >>> logger.debug("This is a debug message")
        2021-09-14T12:00:00Z [my_logger | DEBUG]: This is a debug message
    """

    # Initialize logger object
    logger = logging.getLogger(logger_name)

    # NOTE: Here we cannot use logger.hasHandlers()
    #       Because in runtime, the logger object by default has a stream handler.
    #       This somehow results in logger.hasHandlers() returning True,
    #       but len(logger.handlers) == 0.
    if len(logger.handlers) > 0:
        logger.warning("Logger %s already initialized. Return previous vesrion.", logger_name)
        return logger

    logger.setLevel(file_log_level)
    formatter = logging.Formatter(fmt=env.LOG_FMT, datefmt=env.LOG_DATEFMT)

    # Add stream handler to log to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(streaming_log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_filepath:
        # Check if directory exists
        dirname = os.path.dirname(log_filepath) if os.path.dirname(log_filepath) else "."
        if not os.path.exists(dirname):
            raise FileNotFoundError(f"Directory {dirname} does not exist")

        # Add file handler to log to file
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_filepath,
            when="MIDNIGHT",
            backupCount=7,
        )
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
