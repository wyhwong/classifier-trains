import cv2
import numpy as np
from torchvision import transforms

from .logger import get_logger

LOGGER = get_logger("Preprocessing")
AVAILABLE_INTERPOLATION = ["INTER_NEAREST", "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"]
AVAILABLE_PADDING = ["topLeft", "topRight", "bottomLeft", "bottomRight", None]


def get_transforms(
    width: int,
    height: int,
    interpolation: str,
    maintain_aspect_ratio: bool,
    padding: str,
    mean: list,
    std: list,
    hflip_prob: float,
    vflip_prob: float,
    max_rotate: float,
    centor_crop: bool,
    random_crop: bool,
    gray_scale: bool,
    random_color_augmentation: bool,
) -> dict:
    spatial_augmentation, color_augmentation = [], []
    resize_and_padding = [PilToCV2(), Resize(width, height, interpolation, maintain_aspect_ratio, padding)]
    normalization = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    if hflip_prob > 0:
        LOGGER.debug(f"Spatial augmentation added: {hflip_prob=}.")
        spatial_augmentation.append(transforms.RandomHorizontalFlip(hflip_prob))
    if vflip_prob > 0:
        LOGGER.debug(f"Spatial augmentation added: {vflip_prob=}.")
        spatial_augmentation.append(transforms.RandomVerticalFlip(vflip_prob))
    if max_rotate > 0:
        LOGGER.debug(f"Spatial augmentation added: {max_rotate=}.")
        spatial_augmentation.append(transforms.RandomRotation(max_rotate))
    if centor_crop:
        LOGGER.debug("Spatial augmentation added: center crop.")
        spatial_augmentation.append(transforms.CenterCrop((height, width)))
    if random_crop:
        LOGGER.debug("Spatial augmentation added: random crop.")
        spatial_augmentation.append(transforms.RandomCrop((height, width)))

    if gray_scale:
        LOGGER.debug("Color augmentation added: grayscale.")
        color_augmentation.append(transforms.Grayscale(3))
    if random_color_augmentation:
        brightness, hue = 0.5, 0.3
        LOGGER.debug(f"Color augmentation added: coloer jitter with {brightness=}, {hue=}.")
        color_augmentation.append(transforms.ColorJitter(brightness=brightness, hue=hue))

    LOGGER.info(
        f"Constructing transforms.compose for trainset: \n\t{spatial_augmentation=}, \n\t{color_augmentation=}, \n\t{resize_and_padding=}, \n\t{normalization=}."
    )
    data_transforms = {
        "train": transforms.Compose(spatial_augmentation + color_augmentation + resize_and_padding + normalization),
        "val": transforms.Compose(resize_and_padding + normalization),
    }
    LOGGER.info(f"Constructed transfroms.compose.")
    return data_transforms


class Resize:
    def __init__(self, width: int, height: int, interpolation: str, maintain_aspect_ratio: bool, padding: str) -> None:
        if padding not in AVAILABLE_PADDING:
            raise NotImplementedError(f"Resizing with {padding} padding is invalid or not implemented.")
        if interpolation.upper() not in AVAILABLE_INTERPOLATION:
            raise NotImplementedError(f"Resizing with {interpolation} interpolation is invalid or not implemented.")
        self.input_width = width
        self.input_height = height
        self.dim = (width, height)
        self.interpolation = getattr(cv2, interpolation.upper())
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.padding = padding
        LOGGER.debug(
            f"Resize layer initialized: {width=}, {height=}, {interpolation=}, {maintain_aspect_ratio=}, {padding=}."
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.maintain_aspect_ratio:
            return cv2.resize(image, self.dim, interpolation=self.interpolation)
        else:
            image_height, image_width, _ = image.shape
            if image_height / self.input_height < image_width / self.input_width:
                image_height = int(image_height * (self.input_width / image_width))
                image_width = self.input_width
            else:
                image_width = int(image_width * (self.input_height / image_height))
                image_height = self.input_height
            image = cv2.resize(image, (image_width, image_height), interpolation=self.interpolation)
            output_image = np.zeros((self.input_height, self.input_width, 3), dtype=float)
            if self.padding == "bottomRight":
                output_image[
                    :image_height,
                    :image_width,
                ] = image
            elif self.padding == "bottomLeft":
                output_image[
                    :image_height,
                    self.input_width - image_width :,
                ] = image
            elif self.padding == "topLeft":
                output_image[
                    self.input_height - image_height :,
                    :image_width,
                ] = image
            elif self.padding == "topRight":
                output_image[
                    self.input_height - image_height :,
                    self.input_width - image_width :,
                ] = image
            elif self.padding == "center":
                left = int((self.input_width - image_width) / 2)
                top = int((self.input_height - image_height) / 2)
                output_image[
                    top : top + image_height,
                    left : left + image_width,
                ] = image
            return output_image


class PilToCV2:
    def __init__(self):
        LOGGER.debug(f"PilToCV2 layer initialized.")

    def __call__(self, image):
        return np.array(image)


class Denormalize:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.denormalize = transforms.Normalize(-1 * mean / std, 1 / std)
        LOGGER.debug(f"Denormalize layer initialized.")

    def __call__(self, image):
        return self.denormalize(image)
