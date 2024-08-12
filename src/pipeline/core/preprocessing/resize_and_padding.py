import cv2
import numpy as np
import PIL
from torch import nn

import pipeline.logger
from pipeline.schemas import constants


local_logger = pipeline.logger.get_logger(__name__)


def get_resize_and_padding_transforms(
    width: int,
    height: int,
    interpolation: constants.InterpolationType,
    padding: constants.PaddingType,
    maintain_aspect_ratio: bool,
) -> list[nn.Module]:
    """
    Get the resize and padding transforms.

    Args:
    -----
        width (int):
            The width of the input image.

        height (int):
            The height of the input image.

        interpolation (constants.InterpolationType):
            The interpolation method to use for resizing.

        padding (constants.PaddingType):
            The type of padding to apply to the image.

        maintain_aspect_ratio (bool):
            Whether to maintain the aspect ratio of the image during resizing.

    Returns:
    -----
        resize_and_padding (list[nn.Module]):
            The list of resize and padding transforms.
    """

    resize_and_padding = [
        PilToCV2(),
        Resize(width, height, maintain_aspect_ratio, interpolation, padding),
    ]
    return resize_and_padding


class Resize:
    """
    Resize the image to the given width and height.
    """

    def __init__(
        self,
        width: int,
        height: int,
        maintain_aspect_ratio: bool,
        interpolation: constants.InterpolationType,
        padding: constants.PaddingType,
    ) -> None:
        """
        Initialize the Resize layer.

        Args:
        -----
            width (int):
                The desired width of the input image.

            height (int):
                The desired height of the input image.

            maintain_aspect_ratio (bool):
                Whether to maintain the aspect ratio of the image during resizing.

            interpolation (constants.InterpolationType):
                The interpolation method to use for resizing.

            padding (constants.PaddingType):
                The type of padding to apply to the image.

        Returns:
        -----
            None
        """

        self.__w = width
        self.__h = height
        self.__dim = (width, height)
        self.__interpolation = getattr(cv2, interpolation.value.upper())
        self.__maintain_aspect_ratio = maintain_aspect_ratio
        self.__padding = padding
        local_logger.debug(
            "Resize layer initialized: %.2f, %.2f, %s, %s, %s.",
            width,
            height,
            interpolation,
            maintain_aspect_ratio,
            padding,
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Resize and pad an image according to the specified parameters.

        Args:
        -----
            image (np.ndarray):
                The input image to be processed.

        Returns:
        -----
            output_image (np.ndarray):
                The resized and padded image.
        """

        if not self.__maintain_aspect_ratio:
            return cv2.resize(image, self.__dim, interpolation=self.__interpolation)

        image_height, image_width, _ = image.shape
        # Resize the image to fit the input size while maintaining the aspect ratio.
        if image_height / self.__h < image_width / self.__w:
            image_height = int(image_height * (self.__w / image_width))
            image_width = self.__w
        else:
            image_width = int(image_width * (self.__h / image_height))
            image_height = self.__h

        image = cv2.resize(image, (image_width, image_height), interpolation=self.__interpolation)
        output_image = np.zeros((self.__h, self.__w, 3), dtype=float)

        if self.__padding is constants.PaddingType.BOTTOMRIGHT:
            output_image[
                :image_height,
                :image_width,
            ] = image

        elif self.__padding is constants.PaddingType.BOTTOMLEFT:
            output_image[
                :image_height,
                self.__w - image_width :,
            ] = image

        elif self.__padding is constants.PaddingType.TOPLEFT:
            output_image[
                self.__h - image_height :,
                :image_width,
            ] = image

        elif self.__padding is constants.PaddingType.TOPRIGHT:
            output_image[
                self.__h - image_height :,
                self.__w - image_width :,
            ] = image

        elif self.__padding is constants.PaddingType.CENTER:
            left = int((self.__w - image_width) / 2)
            top = int((self.__h - image_height) / 2)
            output_image[
                top : top + image_height,
                left : left + image_width,
            ] = image

        return output_image


class PilToCV2:
    """
    Convert the PIL image to cv2 image.
    """

    def __init__(self) -> None:
        """
        Initializes the PilToCV2 layer.
        """

        local_logger.debug("PilToCV2 layer initialized.")

    def __call__(self, image: PIL.Image) -> np.ndarray:
        """
        Convert a PIL image to a NumPy array.

        Args:
        -----
            image (PIL.Image):
                The input image.

        Returns:
        -----
            output_image (np.ndarray):
                The converted image as a NumPy array.
        """

        return np.array(image)
