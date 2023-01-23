import os
import yaml
import logging


def getLogger(logger_name: str) -> logging.Logger:
    format = "%(asctime)s [%(name)s | %(levelname)s]: %(message)s"
    datefmt = '%Y-%m-%d %H:%M:%S'
    logger = logging.getLogger(logger_name)
    level = int(os.getenv("LOGGER_LEVEL"))
    logging.basicConfig(level=level, format=format, datefmt=datefmt)
    logger.debug(f"Logger started, level={level}")
    return logger


LOGGER = getLogger("Common")


def getConfig() -> dict:
    LOGGER.info(f"Reading config/config.yml")
    with open("config/config.yml", "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    LOGGER.debug(f"Config: {config}")
    return config


def saveDictAsYml(path:str, inputDict:dict) -> None:
    LOGGER.debug(f"Saving dict: {inputDict}")
    with open(path, 'w') as file:
        yaml.dump(inputDict, file)
    LOGGER.info(f"Saved config at {path}")


def checkAndCreateDir(directory:str) -> bool:
    exist = os.path.isdir(directory)
    LOGGER.debug(f"{directory} exists: {exist}")
    if not exist:
        LOGGER.info(f"{directory} does not exist, creating one.")
        os.mkdir(directory)
    return exist
