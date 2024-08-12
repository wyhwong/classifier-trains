import torchvision
from torch import nn

import pipeline.logger
import pipeline.schemas.constants


local_logger = pipeline.logger.get_logger(__name__)


def get_spatial_transforms(
    width: int,
    height: int,
    hflip_prob: float,
    vflip_prob: float,
    max_rotate_in_degree: float,
    allow_centor_crop: bool,
    allow_random_crop: bool,
) -> list[nn.Module]:
    """
    Get the spatial augmentation transforms.

    Args:
    -----
        width (int):
            The width of the input image.

        height (int):
            The height of the input image.

        hflip_prob (float):
            The probability of horizontal flipping.

        vflip_prob (float):
            The probability of vertical flipping.

        max_rotate_in_degree (float):
            The maximum rotation angle.

        allow_centor_crop (bool):
            Whether to apply center cropping.

        allow_random_crop (bool):
            Whether to apply random cropping.

    Returns:
    -----
        spatial_augmentation (list[nn.Module]):
            The list of spatial augmentation transforms.
    """

    spatial_augmentation = []
    if hflip_prob > 0:
        local_logger.debug("Spatial augmentation added: hflip_prob=%.2f", hflip_prob)
        spatial_augmentation.append(torchvision.transforms.RandomHorizontalFlip(hflip_prob))

    if vflip_prob > 0:
        local_logger.debug("Spatial augmentation added: vflip_prob=%.2f", vflip_prob)
        spatial_augmentation.append(torchvision.transforms.RandomVerticalFlip(vflip_prob))

    if max_rotate_in_degree > 0:
        local_logger.debug("Spatial augmentation added: max_rotate=%.2f", max_rotate_in_degree)
        spatial_augmentation.append(torchvision.transforms.RandomRotation(max_rotate_in_degree))

    if allow_centor_crop:
        local_logger.debug("Spatial augmentation added: center crop.")
        spatial_augmentation.append(torchvision.transforms.CenterCrop((height, width)))

    if allow_random_crop:
        local_logger.debug("Spatial augmentation added: random crop.")
        spatial_augmentation.append(torchvision.transforms.RandomCrop((height, width)))

    return spatial_augmentation


def get_color_transforms(allow_gray_scale: bool, allow_random_color: bool) -> list[nn.Module]:
    """
    Get the color augmentation transforms.

    Args:
    -----
        allow_gray_scale (bool):
            Whether to apply grayscale transformation.

        allow_random_color (bool):
            Whether to apply random color augmentation.

    Returns:
    -----
        color_augmentation (list[nn.Module]):
            The list of color augmentation transforms.
    """

    color_augmentation = []
    if allow_gray_scale:
        local_logger.debug("Color augmentation added: grayscale.")
        color_augmentation.append(torchvision.transforms.Grayscale(3))

    if allow_random_color:
        brightness, hue = 0.5, 0.3
        local_logger.debug(
            "Color augmentation added: coloer jitter with brightness=%.2f, hue=%.2f",
            brightness,
            hue,
        )
        color_augmentation.append(torchvision.transforms.ColorJitter(brightness=brightness, hue=hue))

    return color_augmentation
