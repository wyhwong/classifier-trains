import os
from typing import Any

import yaml

from classifier_trains.utils import logger


local_logger = logger.get_logger(__name__)


def check_and_create_dir(dirpath: str) -> None:
    """Check if the directory exists, if not create it

    Args:
        dirpath (str): The directory path
    """

    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)

    local_logger.info("Created directory: %s", dirpath)


def load_yml(filepath: str) -> dict[str, Any]:
    """Load a yml file

    Args:
        filepath (str): The path to the yml file

    Returns:
        dict[str, Any]: The content of the yml file
    """

    local_logger.info("Read yml: %s", filepath)
    with open(filepath, mode="r", encoding="utf-8") as file:
        content = yaml.load(file, Loader=yaml.SafeLoader)

    local_logger.debug("Yml content: %s", content)
    return content


def save_as_yml(filepath: str, content: dict[str, Any]) -> None:
    """Save content as yml

    Args:
        filepath (str): The path to save the yml file
        content (dict[str, Any]): The content to save
    """

    with open(filepath, mode="w", encoding="utf-8") as file:
        yaml.dump(content, file)

    local_logger.info("Saved config at %s", filepath)
