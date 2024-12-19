from typing import Optional

import torch
import torchvision
import torchvision.transforms.functional
from torch import nn

from classifier_trains.schemas import constants
from classifier_trains.utils import logger


local_logger = logger.get_logger(__name__)


def get_resize_and_padding_transforms(
    width: int,
    height: int,
    interpolation: constants.InterpolationType,
    padding: Optional[constants.PaddingType] = None,
    maintain_aspect_ratio: bool = False,
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

    resize_and_padding: list[nn.Module] = [
        ResizeAndPadding(width, height, interpolation, padding, maintain_aspect_ratio),
    ]
    return resize_and_padding


class ResizeAndPadding(nn.Module):
    """Class to resize the image to the specified dimensions."""

    def __init__(
        self,
        width: int,
        height: int,
        interpolation: constants.InterpolationType,
        padding: Optional[constants.PaddingType] = None,
        maintain_aspect_ratio: bool = False,
    ) -> None:
        """Initialize the Resize layer.

        Args:
            width (int): The width of the image.
            height (int): The height of the image.
            interpolation (constants.InterpolationType): The interpolation type.
            padding (constants.PaddingType): The padding type.
            maintain_aspect_ratio (bool): Whether to maintain the aspect ratio.
        """

        super().__init__()

        self._w = width
        self._h = height

        self._padding = padding
        self._interpolation = getattr(torchvision.transforms.InterpolationMode, interpolation.value.upper())
        self._maintain_aspect_ratio = maintain_aspect_ratio

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Resize the image to the specified dimensions.

        Args:
            image (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The resized image.
        """

        if not self._maintain_aspect_ratio:
            return torchvision.transforms.functional.resize(image, (self._h, self._w), self._interpolation)

        # Resize the image while maintaining the aspect ratio
        h, w = image.shape[-2:]

        if h / self._h < w / self._w:
            h, w = int(h * (self._w / w)), self._w
        else:
            w, h = int(w * (self._h / h)), self._h

        image = torchvision.transforms.functional.resize(image, (h, w), self._interpolation)
        output_image = torch.zeros((3, self._h, self._w), dtype=image.dtype)

        if self._padding is constants.PaddingType.TOPLEFT:
            output_image[:, :h, :w] = image
        elif self._padding is constants.PaddingType.TOPRIGHT:
            output_image[:, :h, -w:] = image

        elif self._padding is constants.PaddingType.BOTTOMLEFT:
            output_image[:, -h:, :w] = image

        elif self._padding is constants.PaddingType.BOTTOMRIGHT:
            output_image[:, -h:, -w:] = image

        elif self._padding is constants.PaddingType.CENTER:
            output_image[
                :, (self._h - h) // 2 : (self._h - h) // 2 + h, (self._w - w) // 2 : (self._w - w) // 2 + w
            ] = image

        else:
            local_logger.error("Invalid padding type: %s", self._padding)
            raise ValueError(f"Invalid padding type: {self._padding}")

        return output_image
