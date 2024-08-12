import numpy as np
import torchvision

import pipeline.logger


local_logger = pipeline.logger.get_logger(__name__)


class Denormalize:
    """
    Denormalize the image with the given mean and standard deviation.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        """
        Initialize the Preprocessing object.

        Args:
        -----
            mean (np.ndarray):
                Mean values for normalization.

            std (np.ndarray):
                Standard deviation values for normalization.
        """

        self._denormalize = torchvision.transforms.Normalize(-1 * mean / std, 1 / std)
        local_logger.debug("Denormalize layer initialized.")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing operations on the input image.

        Args:
        -----
            image (np.ndarray):
                The input image.

        Returns:
        -----
            output_image (np.ndarray):
                The preprocessed image.
        """

        return self._denormalize(image)
