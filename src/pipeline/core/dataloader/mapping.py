from typing import Optional

import torchvision

import pipeline.core.utils
import pipeline.logger


local_logger = pipeline.logger.get_logger(__name__)


def save_class_mapping(
    dataset: torchvision.datasets.ImageFolder,
    savepath: Optional[str],
) -> None:
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
        none
    """

    mapping = dataset.class_to_idx
    local_logger.info("Reading class mapping in the dataset: %s.", mapping)

    if savepath:
        pipeline.core.utils.save_as_yml(savepath, mapping)
        local_logger.info("Saved class mapping to %s.", savepath)
