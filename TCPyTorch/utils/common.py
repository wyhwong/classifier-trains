import os
import yaml
from .logger import get_logger

LOGGER = get_logger("utils/common")


def get_config() -> dict:
    return load_yml("/train.yml")


def load_yml(filepath: str) -> dict:
    LOGGER.debug("Read yml: %s", filepath)
    with open(filepath, "r") as file:
        yml_content = yaml.load(file, Loader=yaml.SafeLoader)
    return yml_content


def save_dict_as_yml(filepath: str, input_dict: dict) -> None:
    with open(filepath, "w") as file:
        yaml.dump(input_dict, file)
    LOGGER.info("Saved config at %s", filepath)


def check_and_create_dir(dirpath: str) -> bool:
    exist = os.path.isdir(dirpath)
    if not exist:
        LOGGER.info("%s does not exist, created", dirpath)
        os.mkdir(dirpath)
    return exist
