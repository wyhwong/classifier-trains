import cv2
import numpy as np
import PIL
import torchvision

import utils.logger
import utils.constants

LOGGER = utils.logger.get_logger("utils/preprocessing")


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
) -> dict[str, torchvision.transforms.Compose]:
    """
    Get the data transforms for the trainset and valset.

    Parameters
    ----------
    width: int, required, the width of the image.
    height: int, required, the height of the image.
    interpolation: str, required, the interpolation method.
    maintain_aspect_ratio: bool, required, whether to maintain the aspect ratio or not.
    padding: str, required, the padding method.
    mean: list, required, the mean of the dataset.
    std: list, required, the standard deviation of the dataset.
    hflip_prob: float, required, the probability of horizontal flipping.
    vflip_prob: float, required, the probability of vertical flipping.
    max_rotate: float, required, the maximum rotation angle.
    centor_crop: bool, required, whether to use center crop or not.

    Returns
    -------
    data_transforms: dict, the data transforms.
    """
    spatial_augmentation, color_augmentation = [], []
    resize_and_padding = [
        PilToCV2(),
        Resize(width, height, interpolation, maintain_aspect_ratio, padding),
    ]
    normalization = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ]

    if hflip_prob > 0:
        LOGGER.debug("Spatial augmentation added: hflip_prob=%.2f", hflip_prob)
        spatial_augmentation.append(
            torchvision.transforms.RandomHorizontalFlip(hflip_prob)
        )
    if vflip_prob > 0:
        LOGGER.debug("Spatial augmentation added: vflip_prob=%.2f", vflip_prob)
        spatial_augmentation.append(
            torchvision.transforms.RandomVerticalFlip(vflip_prob)
        )
    if max_rotate > 0:
        LOGGER.debug("Spatial augmentation added: max_rotate=%.2f", max_rotate)
        spatial_augmentation.append(torchvision.transforms.RandomRotation(max_rotate))
    if centor_crop:
        LOGGER.debug("Spatial augmentation added: center crop.")
        spatial_augmentation.append(torchvision.transforms.CenterCrop((height, width)))
    if random_crop:
        LOGGER.debug("Spatial augmentation added: random crop.")
        spatial_augmentation.append(torchvision.transforms.RandomCrop((height, width)))

    if gray_scale:
        LOGGER.debug("Color augmentation added: grayscale.")
        color_augmentation.append(torchvision.transforms.Grayscale(3))
    if random_color_augmentation:
        brightness, hue = 0.5, 0.3
        LOGGER.debug(
            "Color augmentation added: coloer jitter with brightness=%.2f, hue=%.2f",
            brightness,
            hue,
        )
        color_augmentation.append(
            torchvision.transforms.ColorJitter(brightness=brightness, hue=hue)
        )

    construction_message = "Constructing torchvision.transforms.compose for trainset: "
    for transform in [
        spatial_augmentation,
        color_augmentation,
        resize_and_padding,
        normalization,
    ]:
        construction_message += f"\n\t{transform}"
    LOGGER.info(construction_message)

    data_transforms = {
        "train": torchvision.transforms.Compose(
            spatial_augmentation
            + color_augmentation
            + resize_and_padding
            + normalization
        ),
        "val": torchvision.transforms.Compose(resize_and_padding + normalization),
    }
    LOGGER.info("Constructed torchvision.transforms.compose.")
    return data_transforms


class Resize:
    """
    Resize the image to the given width and height.
    """

    def __init__(
        self,
        width: int,
        height: int,
        interpolation: str,
        maintain_aspect_ratio: bool,
        padding: str,
    ) -> None:
        """
        Initialize the image resizer.

        Parameters
        ----------
        width: int, required, the width of the image.
        height: int, required, the height of the image.
        interpolation: str, required, the interpolation method.
        maintain_aspect_ratio: bool, required, whether to maintain the aspect ratio or not.
        padding: str, required, the padding method.

        Returns
        -------
        None
        """
        if padding not in utils.constants.AVAILABLE_PADDING:
            raise NotImplementedError(
                f"Resizing with {padding} padding is invalid or not implemented."
            )
        if interpolation.upper() not in utils.constants.AVAILABLE_INTERPOLATION:
            raise NotImplementedError(
                f"Resizing with {interpolation} interpolation is invalid or not implemented."
            )
        self.input_width = width
        self.input_height = height
        self.dim = (width, height)
        self.interpolation = getattr(cv2, interpolation.upper())
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.padding = padding
        LOGGER.debug(
            "Resize layer initialized: %.2f, %.2f, %s, %b, %s.",
            width,
            height,
            interpolation,
            maintain_aspect_ratio,
            padding,
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Resize the image to the given width and height.

        Parameters
        ----------
        image: np.ndarray, required, the image to be resized.

        Returns
        -------
        image: np.ndarray, the resized image.
        """
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
            image = cv2.resize(
                image, (image_width, image_height), interpolation=self.interpolation
            )
            output_image = np.zeros(
                (self.input_height, self.input_width, 3), dtype=float
            )
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
    """
    Convert the PIL image to cv2 image.
    """

    def __init__(self) -> None:
        """
        Initialize the PIL to cv2 converter.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        LOGGER.debug("PilToCV2 layer initialized.")

    def __call__(self, image: PIL.Image) -> np.ndarray:
        """
        Convert the PIL image to cv2 image.

        Parameters
        ----------
        image: PIL.Image, required, the PIL image to be converted.

        Returns
        -------
        image: np.ndarray, the converted cv2 image.
        """
        return np.array(image)


class Denormalize:
    """
    Denormalize the image with the given mean and standard deviation.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        """
        Initialize the image denormalizer.

        Parameters
        ----------
        mean: np.ndarray, required, the mean of the dataset.
        std: np.ndarray, required, the standard deviation of the dataset.

        Returns
        -------
        None
        """
        self.denormalize = torchvision.transforms.Normalize(-1 * mean / std, 1 / std)
        LOGGER.debug("Denormalize layer initialized.")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Denormalize the image with the given mean and standard deviation.

        Parameters
        ----------
        image: np.ndarray, required, the image to be denormalized.

        Returns
        -------
        image: np.ndarray, the denormalized image.
        """
        return self.denormalize(image)
