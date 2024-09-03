import cv2
import numpy as np
from PIL import Image
from torch import nn

from pipeline.schemas import constants
from pipeline.utils import logger


local_logger = logger.get_logger(__name__)


def get_resize_and_padding_transforms(
    width: int,
    height: int,
    interpolation: constants.InterpolationType,
    padding: constants.PaddingType,
    maintain_aspect_ratio: bool,
) -> list[nn.Module]:
    """Function to get the resize and padding transforms.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        interpolation (constants.InterpolationType): The interpolation type.
        padding (constants.PaddingType): The padding type.
        maintain_aspect_ratio (bool): Whether to maintain the aspect ratio.

    Returns:
        list[nn.Module]: The list of resize and padding transforms.
    """

    resize_and_padding = [
        PilToCV2(),
        Resize(width, height, maintain_aspect_ratio, interpolation, padding),
    ]
    return resize_and_padding


class Resize(nn.Module):
    """Class to resize the image to the specified dimensions."""

    def __init__(
        self,
        width: int,
        height: int,
        maintain_aspect_ratio: bool,
        interpolation: constants.InterpolationType,
        padding: constants.PaddingType,
    ) -> None:
        """Initialize the Resize layer.

        Args:
            width (int): The width of the image.
            height (int): The height of the image.
            maintain_aspect_ratio (bool): Whether to maintain the aspect ratio.
            interpolation (constants.InterpolationType): The interpolation type.
            padding (constants.PaddingType): The padding type.
        """

        super().__init__()

        self.__w = width
        self.__h = height
        self.__dim = (width, height)
        self.__interpolation = getattr(cv2, interpolation.value.upper())
        self.__maintain_aspect_ratio = maintain_aspect_ratio
        self.__padding = padding

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Resize the image to the specified dimensions.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The resized image.
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


class PilToCV2(nn.Module):
    """Convert the PIL image to cv2 image."""

    def __call__(self, image: Image.Image) -> np.ndarray:
        """Convert the PIL image to cv2 image.

        Args:
            image (Image.Image): The PIL image.

        Returns:
            np.ndarray: The cv2 image.
        """

        return np.array(image)
