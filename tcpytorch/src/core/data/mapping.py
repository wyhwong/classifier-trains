from typing import Optional

import torchvision

import core.utils
import logger


local_logger = logger.get_logger(__name__)


def get_class_mapping(
    dataset: torchvision.datasets.ImageFolder,
    savepath: Optional[str],
) -> dict[str, str]:
    """
    Get class mapping from the dataset.

    Args:
    -----
        dataset: torchvision.datasets.ImageFolder
            Dataset to get class mapping.

        savepath: str, optional
            Path to save the class mapping.

    Returns:
    --------
        mapping: dict[str, str]
            Class mapping of the dataset.
    """

    mapping = dataset.class_to_idx
    local_logger.info("Reading class mapping in the dataset: %s.", mapping)

    if savepath:
        core.utils.save_as_yml(savepath, mapping)
        local_logger.info("Saved class mapping to %s.", savepath)

    return mapping
