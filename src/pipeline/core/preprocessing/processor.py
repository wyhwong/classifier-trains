import numpy as np
import torch
import torchvision

import pipeline.logger
from pipeline.core.preprocessing.augmentation import get_color_transforms, get_spatial_transforms
from pipeline.core.preprocessing.resize_and_padding import get_resize_and_padding_transforms
from pipeline.schemas import config


local_logger = pipeline.logger.get_logger(__name__)


class Preprocessor:
    """Class to preprocess the input data."""

    def __init__(self, preprocessing_config: config.PreprocessingConfig) -> None:
        """Initialize the Preprocessor object.

        Args:
            preprocessing_config (config.PreprocessingConfig): The preprocessing configuration.
        """

        self.__mean = np.array(preprocessing_config.mean)
        self.__std = np.array(preprocessing_config.std)

        self.__spatial_config = preprocessing_config.spatial_config
        self.__color_config = preprocessing_config.color_config
        self.__resize_config = preprocessing_config.resize_config

        self.__denormalizer = torchvision.transforms.Normalize(
            mean=-1 * self.__mean / self.__std,
            std=1 / self.__std,
        )

        self.__construct_transforms()

    def __construct_transforms(self) -> None:
        """Construct the data transforms."""

        self.__normalization = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.__mean, self.__std),
        ]
        self.__spatial_augmentation = get_spatial_transforms(
            width=self.__resize_config.width,
            height=self.__resize_config.height,
            hflip_prob=self.__spatial_config.hflip_prob,
            vflip_prob=self.__spatial_config.vflip_prob,
            max_rotate_in_degree=self.__spatial_config.max_rotate_in_degree,
            allow_center_crop=self.__spatial_config.allow_center_crop,
            allow_random_crop=self.__spatial_config.allow_random_crop,
        )
        self.__color_augmentation = get_color_transforms(
            allow_gray_scale=self.__color_config.allow_gray_scale,
            allow_random_color=self.__color_config.allow_random_color,
        )
        self.__resize_and_padding = get_resize_and_padding_transforms(
            width=self.__resize_config.width,
            height=self.__resize_config.height,
            interpolation=self.__resize_config.interpolation,
            padding=self.__resize_config.padding,
            maintain_aspect_ratio=self.__resize_config.maintain_aspect_ratio,
        )

    def __call__(self, image: np.ndarray, is_augmented: bool) -> torch.Tensor:
        """Preprocess the input image.

        Args:
            image (np.ndarray): The input image.
            is_augmented (bool): Whether the image is augmented or not.

        Returns:
            torch.Tensor: The preprocessed image.
        """

        if is_augmented:
            layers = self.__spatial_augmentation + self.__color_augmentation
        else:
            layers = []
        layers += self.__resize_and_padding + self.__normalization

        for layer in layers:
            image = layer(image)

        return image

    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """Denormalize the input image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The denormalized image.
        """

        return self.__denormalizer(image)

    def get_training_transforms(self) -> torchvision.transforms.Compose:
        """Get the training transforms.

        Returns:
            torchvision.transforms.Compose: The training transforms.
        """

        return torchvision.transforms.Compose(
            self.__spatial_augmentation + self.__color_augmentation + self.__resize_and_padding + self.__normalization
        )

    def get_validation_transforms(self) -> torchvision.transforms.Compose:
        """Get the validation transforms.

        Returns:
            torchvision.transforms.Compose: The validation transforms.
        """

        return torchvision.transforms.Compose(self.__resize_and_padding + self.__normalization)
