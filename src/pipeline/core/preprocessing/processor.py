import numpy as np
import torchvision
from torch import nn

import pipeline.logger
from pipeline.core.preprocessing.augmentation import get_color_transforms, get_spatial_transforms
from pipeline.core.preprocessing.resize_and_padding import get_resize_and_padding_transforms
from pipeline.schemas import config, constants


local_logger = pipeline.logger.get_logger(__name__)


class Preprocessor:
    """Class to preprocess the input data."""

    def __init__(self, preprocessing_config: config.PreprocessingConfig) -> None:
        """Initialize the Preprocessor object.

        Args:
            preprocessing_config (config.PreprocessingConfig): The preprocessing configuration.
        """

        self.__mean = preprocessing_config.mean
        self.__std = preprocessing_config.std

        self.__spatial_config = preprocessing_config.spatial_augmentation
        self.__color_config = preprocessing_config.color_augmentation
        self.__resize_config = preprocessing_config.resize_and_padding

        self.__denormalizer = torchvision.transforms.Normalize(
            mean=-1 * self.__mean / self.__std,
            std=1 / self.__std,
        )

    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """Denormalize the input image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The denormalized image.
        """

        return self.__denormalizer(image)

    def construct_transforms_compose(self) -> dict[constants.Phase, torchvision.transforms.Compose]:
        """Construct the data transforms."""

        spatial_augmentation = get_spatial_transforms(
            width=self.__resize_config.width,
            height=self.__resize_config.height,
            hflip_prob=self.__spatial_config.hflip_prob,
            vflip_prob=self.__spatial_config.vflip_prob,
            max_rotate_in_degree=self.__spatial_config.max_rotate_in_degree,
            allow_centor_crop=self.__spatial_config.allow_centor_crop,
            allow_random_crop=self.__spatial_config.allow_random_crop,
        )
        color_augmentation = get_color_transforms(
            allow_gray_scale=self.__color_config.allow_gray_scale,
            allow_random_color=self.__color_config.allow_random_color,
        )
        resize_and_padding = get_resize_and_padding_transforms(
            width=self.__resize_config.width,
            height=self.__resize_config.height,
            interpolation=self.__resize_config.interpolation,
            padding=self.__resize_config.padding,
            maintain_aspect_ratio=self.__resize_config.maintain_aspect_ratio,
        )

        return self.combine_transforms(
            spatial_augmentation=spatial_augmentation,
            color_augmentation=color_augmentation,
            resize_and_padding=resize_and_padding,
            mean=self.__mean,
            std=self.__std,
        )

    @staticmethod
    def combine_transforms(
        spatial_augmentation: list[nn.Module],
        color_augmentation: list[nn.Module],
        resize_and_padding: list[nn.Module],
        mean: list[float],
        std: list[float],
    ) -> dict[constants.Phase, torchvision.transforms.Compose]:
        """Combine the transforms.

        Args:
            spatial_augmentation (list[nn.Module]): The spatial augmentation transforms.
            color_augmentation (list[nn.Module]): The color augmentation transforms.
            resize_and_padding (list[nn.Module]): The resize and padding transforms.
            mean (list[float]): The mean values for normalization.
            std (list[float]): The standard deviation values for normalization.

        Returns:
            dict[constants.Phase, torchvision.transforms.Compose]: The data transforms
        """

        normalization = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]

        data_transforms = {
            constants.Phase.TRAINING: torchvision.transforms.Compose(
                spatial_augmentation + color_augmentation + resize_and_padding + normalization
            ),
            constants.Phase.VALIDATION: torchvision.transforms.Compose(resize_and_padding + normalization),
        }

        return data_transforms
