import os
import yaml
from .logger import get_logger

LOGGER = get_logger("Common")


def get_config() -> dict:
    return load_yml("configs/config.yml")


def load_yml(ymlpath: str) -> dict:
    LOGGER.info(f"Read yml: {ymlpath}")
    with open("config/config.yml", "r") as file:
        yml_content = yaml.load(file, Loader=yaml.SafeLoader)
    return yml_content


def save_dict_as_yml(ymlpath: str, input_dict: dict) -> None:
    LOGGER.debug(f"Saving dict: {input_dict}")
    with open(ymlpath, "w") as file:
        yaml.dump(input_dict, file)
    LOGGER.info(f"Saved config at {ymlpath}")


def check_and_create_dir(directory: str) -> bool:
    exist = os.path.isdir(directory)
    LOGGER.debug(f"{directory} exists: {exist}")
    if not exist:
        LOGGER.info(f"{directory} does not exist, creating one.")
        os.mkdir(directory)
    return exist
