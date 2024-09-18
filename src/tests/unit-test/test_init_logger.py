import logging
import os
import sys

import pytest

import classifier_trains.utils.env
from classifier_trains.utils.logger import get_logger


def test_get_logger_1() -> None:
    """Test the get_logger function without log_filepath.

    Expected behavior:
        - The function should return a logger object with the specified name and log levels.
        - The logger object should have one stream handler with the specified log level and formatter.
        - The stream handler should log to sys.stdout.
    """

    # Test the get_logger function
    logger = get_logger(
        logger_name="test_logger_1",
        log_filepath=None,
    )
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger_1"
    assert logger.level == classifier_trains.utils.env.FILE_LOG_LEVEL
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].level == classifier_trains.utils.env.STREAMING_LOG_LEVEL
    assert logger.handlers[0].formatter._fmt == classifier_trains.utils.env.LOG_FMT
    assert logger.handlers[0].formatter.datefmt == classifier_trains.utils.env.LOG_DATEFMT
    assert logger.handlers[0].stream == sys.stdout
    assert logger.hasHandlers()


def test_get_logger_2() -> None:
    """Test the get_logger function with log_filepath.

    Expected behavior:
        - The function should return a logger object with the specified name and log levels.
        - The logger object should have two handlers: a stream handler and a file handler.
        - The stream handler should have the specified log level and formatter.
        - The stream handler should log to sys.stdout.
        - The file handler should have the specified log level, formatter, and log file settings.
    """

    logger = get_logger(logger_name="test_logger_2")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger_2"
    assert logger.level == classifier_trains.utils.env.FILE_LOG_LEVEL
    assert len(logger.handlers) == 2
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].level == classifier_trains.utils.env.STREAMING_LOG_LEVEL
    assert logger.handlers[0].formatter._fmt == classifier_trains.utils.env.LOG_FMT
    assert logger.handlers[0].formatter.datefmt == classifier_trains.utils.env.LOG_DATEFMT
    assert logger.handlers[0].stream == sys.stdout
    assert isinstance(logger.handlers[1], logging.handlers.TimedRotatingFileHandler)
    assert logger.handlers[1].level == classifier_trains.utils.env.FILE_LOG_LEVEL
    assert logger.handlers[1].formatter._fmt == classifier_trains.utils.env.LOG_FMT
    assert logger.handlers[1].formatter.datefmt == classifier_trains.utils.env.LOG_DATEFMT
    assert logger.handlers[1].baseFilename == os.path.abspath(classifier_trains.utils.env.LOG_FILEPATH)
    assert logger.handlers[1].when == "MIDNIGHT"
    assert logger.handlers[1].backupCount == 7
    assert logger.hasHandlers()

    # Clean up
    os.remove(classifier_trains.utils.env.LOG_FILEPATH)


def test_get_logger_3() -> None:
    """Test the get_logger function with a nonexistent log file.

    Expected behavior:
        - The function should raise a FileNotFoundError.
    """

    with pytest.raises(FileNotFoundError):
        get_logger(
            logger_name="test_logger_3",
            log_filepath="./nonexistent/test.log",
        )


def test_get_logger_4() -> None:
    """Test the get_logger function with a previously created logger.

    Expected behavior:
        - The function should return the existing logger object.
    """

    logger = get_logger(
        logger_name="test_logger_4",
        log_filepath=None,
    )
    assert len(logger.handlers) == 1

    # If the logger already exists, the function should return the existing logger.
    logger = get_logger(logger_name="test_logger_4")
    assert len(logger.handlers) == 1
