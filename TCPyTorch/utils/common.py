import os
import yaml
import utils.logger

LOGGER = utils.logger.get_logger("utils/common")


def get_config() -> dict:
    """
    Get the config from the config.yml file.

    Returns
    -------
    config: dict, training config.
    """
    return load_yml("/train.yml")


def load_yml(filepath: str) -> dict:
    """
    Load the yml file from the given filepath.

    Parameters
    ----------
    filepath: str, required, the filepath of the yml file.

    Returns
    -------
    yml_content: dict, the content of the yml file.
    """
    LOGGER.debug("Read yml: %s", filepath)
    with open(filepath, "r") as file:
        yml_content = yaml.load(file, Loader=yaml.SafeLoader)
    return yml_content


def save_dict_as_yml(filepath: str, input_dict: dict) -> None:
    """
    Save the input dict as a yml file.

    Parameters
    ----------
    filepath: str, required, the filepath of the yml file.
    input_dict: dict, required, the dict to be saved.

    Returns
    -------
    None
    """
    with open(filepath, "w") as file:
        yaml.dump(input_dict, file)
    LOGGER.info("Saved config at %s", filepath)


def check_and_create_dir(dirpath: str) -> bool:
    """
    Check if the given dirpath exists, if not, create it.

    Parameters
    ----------
    dirpath: str, required, the dirpath to be checked and created.

    Returns
    -------
    exist: bool, whether the dirpath exists.
    """
    exist = os.path.isdir(dirpath)
    if not exist:
        LOGGER.info("%s does not exist, created", dirpath)
        os.mkdir(dirpath)
    return exist
