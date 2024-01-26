import cv2
import numpy as np
import PIL
import torchvision

import logger
import schemas.constants


local_logger = logger.get_logger(__name__)


def get_spatial_transforms(
    width: int,
    height: int,
    hflip_prob: float,
    vflip_prob: float,
    max_rotate: float,
    centor_crop: bool,
    random_crop: bool,
) -> list[torchvision.transforms]:
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

        max_rotate (float):
            The maximum rotation angle.

        centor_crop (bool):
            Whether to apply center cropping.

        random_crop (bool):
            Whether to apply random cropping.

    Returns:
    -----
        spatial_augmentation (list[torchvision.transforms]):
            The list of spatial augmentation transforms.
    """

    spatial_augmentation = []
    if hflip_prob > 0:
        local_logger.debug("Spatial augmentation added: hflip_prob=%.2f", hflip_prob)
        spatial_augmentation.append(torchvision.transforms.RandomHorizontalFlip(hflip_prob))

    if vflip_prob > 0:
        local_logger.debug("Spatial augmentation added: vflip_prob=%.2f", vflip_prob)
        spatial_augmentation.append(torchvision.transforms.RandomVerticalFlip(vflip_prob))

    if max_rotate > 0:
        local_logger.debug("Spatial augmentation added: max_rotate=%.2f", max_rotate)
        spatial_augmentation.append(torchvision.transforms.RandomRotation(max_rotate))

    if centor_crop:
        local_logger.debug("Spatial augmentation added: center crop.")
        spatial_augmentation.append(torchvision.transforms.CenterCrop((height, width)))

    if random_crop:
        local_logger.debug("Spatial augmentation added: random crop.")
        spatial_augmentation.append(torchvision.transforms.RandomCrop((height, width)))

    return spatial_augmentation


def get_color_transforms(
    gray_scale: bool,
    random_color_augmentation: bool,
) -> list[torchvision.transforms]:
    """
    Get the color augmentation transforms.

    Args:
    -----
        gray_scale (bool):
            Whether to apply grayscale transformation.

        random_color_augmentation (bool):
            Whether to apply random color augmentation.

    Returns:
    -----
        color_augmentation (list[torchvision.transforms]):
            The list of color augmentation transforms.
    """

    color_augmentation = []
    if gray_scale:
        local_logger.debug("Color augmentation added: grayscale.")
        color_augmentation.append(torchvision.transforms.Grayscale(3))

    if random_color_augmentation:
        brightness, hue = 0.5, 0.3
        local_logger.debug(
            "Color augmentation added: coloer jitter with brightness=%.2f, hue=%.2f",
            brightness,
            hue,
        )
        color_augmentation.append(torchvision.transforms.ColorJitter(brightness=brightness, hue=hue))
    return color_augmentation


def get_resize_and_padding_transforms(
    width: int,
    height: int,
    maintain_aspect_ratio: bool,
    interpolation: schemas.constants.InterpolationType,
    padding: schemas.constants.PaddingType,
) -> list[torchvision.transforms]:
    """
    Get the resize and padding transforms.

    Args:
    -----
        width (int):
            The width of the input image.

        height (int):
            The height of the input image.

        maintain_aspect_ratio (bool):
            Whether to maintain the aspect ratio of the image during resizing.

        interpolation (schemas.constants.InterpolationType):
            The interpolation method to use for resizing.

        padding (schemas.constants.PaddingType):
            The type of padding to apply to the image.

    Returns:
    -----
        resize_and_padding (list[torchvision.transforms]):
            The list of resize and padding transforms.
    """

    resize_and_padding = [
        PilToCV2(),
        Resize(width, height, maintain_aspect_ratio, interpolation, padding),
    ]
    return resize_and_padding


def get_transforms(
    spatial_augmentation: list[torchvision.transforms],
    color_augmentation: list[torchvision.transforms],
    resize_and_padding: list[torchvision.transforms],
    mean: list,
    std: list,
) -> dict[str, torchvision.transforms.Compose]:
    """
    Get the data transforms.

    Args:
    -----
        spatial_augmentation (list[torchvision.transforms]):
            The list of spatial augmentation transforms.

        color_augmentation (list[torchvision.transforms]):
            The list of color augmentation transforms.

        resize_and_padding (list[torchvision.transforms]):
            The list of resize and padding transforms.

        mean (list):
            The mean values for normalization.

        std (list):
            The standard deviation values for normalization.

    Returns:
    -----
        data_transforms (dict[str, torchvision.transforms.Compose]):
            The dictionary of data transforms.
    """

    normalization = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ]
    construction_message = "Constructing torchvision.transforms.compose for trainset: "
    for transform in [
        spatial_augmentation,
        color_augmentation,
        resize_and_padding,
        normalization,
    ]:
        construction_message += f"\n\t{transform}"
    local_logger.info(construction_message)

    data_transforms = {
        schemas.constants.Phase.TRAINING.value: torchvision.transforms.Compose(
            spatial_augmentation + color_augmentation + resize_and_padding + normalization
        ),
        schemas.constants.Phase.VALIDATION.value: torchvision.transforms.Compose(resize_and_padding + normalization),
    }
    local_logger.info("Constructed torchvision.transforms.compose.")
    return data_transforms


class Resize:
    """
    Resize the image to the given width and height.
    """

    def __init__(
        self,
        width: int,
        height: int,
        maintain_aspect_ratio: bool,
        interpolation: schemas.constants.InterpolationType,
        padding: schemas.constants.PaddingType,
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

            interpolation (schemas.constants.InterpolationType):
                The interpolation method to use for resizing.

            padding (schemas.constants.PaddingType):
                The type of padding to apply to the image.

        Returns:
        -----
            None
        """

        self.input_width = width
        self.input_height = height
        self.dim = (width, height)
        self.interpolation = getattr(cv2, interpolation.value.upper())
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.padding = padding
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

        if not self.maintain_aspect_ratio:
            return cv2.resize(image, self.dim, interpolation=self.interpolation)

        image_height, image_width, _ = image.shape
        # Resize the image to fit the input size while maintaining the aspect ratio.
        if image_height / self.input_height < image_width / self.input_width:
            image_height = int(image_height * (self.input_width / image_width))
            image_width = self.input_width
        else:
            image_width = int(image_width * (self.input_height / image_height))
            image_height = self.input_height

        image = cv2.resize(image, (image_width, image_height), interpolation=self.interpolation)
        output_image = np.zeros((self.input_height, self.input_width, 3), dtype=float)

        if self.padding is schemas.constants.PaddingType.BOTTOMRIGHT:
            output_image[
                :image_height,
                :image_width,
            ] = image
        elif self.padding is schemas.constants.PaddingType.BOTTOMLEFT:
            output_image[
                :image_height,
                self.input_width - image_width :,
            ] = image
        elif self.padding is schemas.constants.PaddingType.TOPLEFT:
            output_image[
                self.input_height - image_height :,
                :image_width,
            ] = image
        elif self.padding is schemas.constants.PaddingType.TOPRIGHT:
            output_image[
                self.input_height - image_height :,
                self.input_width - image_width :,
            ] = image
        elif self.padding is schemas.constants.PaddingType.CENTER:
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
