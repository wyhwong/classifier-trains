import torchvision
from torch import nn

from classifier_trains.utils import logger


local_logger = logger.get_logger(__name__)


def get_spatial_transforms(
    width: int,
    height: int,
    hflip_prob: float,
    vflip_prob: float,
    max_rotate_in_degree: float,
    allow_center_crop: bool,
    allow_random_crop: bool,
) -> list[nn.Module]:
    """Get the spatial augmentation transforms.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        hflip_prob (float): The probability of horizontal flip.
        vflip_prob (float): The probability of vertical flip.
        max_rotate_in_degree (float): The maximum rotation in degree.
        allow_center_crop (bool): Whether to allow center crop.
        allow_random_crop (bool): Whether to allow random crop.

    Returns:
        list[nn.Module]: The list of spatial augmentation transforms.
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

    if allow_center_crop:
        local_logger.debug("Spatial augmentation added: center crop.")
        spatial_augmentation.append(torchvision.transforms.CenterCrop((height, width)))

    if allow_random_crop:
        local_logger.debug("Spatial augmentation added: random crop.")
        spatial_augmentation.append(torchvision.transforms.RandomCrop((height, width)))

    return spatial_augmentation


def get_color_transforms(allow_gray_scale: bool, allow_random_color: bool) -> list[nn.Module]:
    """Get the color augmentation transforms.

    Args:
        allow_gray_scale (bool): Whether to allow grayscale.
        allow_random_color (bool): Whether to allow random color.

    Returns:
        list[nn.Module]: The list of color augmentation transforms.
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
