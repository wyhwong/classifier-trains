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

    resize_and_padding: list[nn.Module] = [
        ResizeAndPadding(width, height, maintain_aspect_ratio, interpolation, padding),
    ]
    return resize_and_padding


class ResizeAndPadding(nn.Module):
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

        self.__interpolation = getattr(torchvision.transforms.InterpolationMode, interpolation.value.upper())
        self.__maintain_aspect_ratio = maintain_aspect_ratio
        self.__padding = padding

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Resize the image to the specified dimensions.

        Args:
            image (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The resized image.
        """

        if not self.__maintain_aspect_ratio:
            return torchvision.transforms.functional.resize(image, (self.__h, self.__w), self.__interpolation)

        # Resize the image while maintaining the aspect ratio
        h, w = image.shape[-2:]

        if h / self.__h < w / self.__w:
            h, w = int(h * (self.__w / w)), self.__w
        else:
            w, h = int(w * (self.__h / h)), self.__h

        image = torchvision.transforms.functional.resize(image, (h, w), self.__interpolation)
        output_image = torch.zeros((3, self.__h, self.__w), dtype=image.dtype)

        if self.__padding is constants.PaddingType.TOPLEFT:
            output_image[:, :h, :w] = image
        elif self.__padding is constants.PaddingType.TOPRIGHT:
            output_image[:, :h, -w:] = image

        elif self.__padding is constants.PaddingType.BOTTOMLEFT:
            output_image[:, -h:, :w] = image

        elif self.__padding is constants.PaddingType.BOTTOMRIGHT:
            output_image[:, -h:, -w:] = image

        elif self.__padding is constants.PaddingType.CENTER:
            output_image[
                :, (self.__h - h) // 2 : (self.__h - h) // 2 + h, (self.__w - w) // 2 : (self.__w - w) // 2 + w
            ] = image

        else:
            local_logger.error("Invalid padding type: %s", self.__padding)
            raise ValueError(f"Invalid padding type: {self.__padding}")

        return output_image
