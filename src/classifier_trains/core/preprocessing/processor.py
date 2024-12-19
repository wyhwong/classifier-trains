import numpy as np
import torch
import torchvision

from classifier_trains.core.preprocessing.augmentation import get_color_transforms, get_spatial_transforms
from classifier_trains.core.preprocessing.resize_and_padding import get_resize_and_padding_transforms
from classifier_trains.schemas import config
from classifier_trains.utils import logger


local_logger = logger.get_logger(__name__)


class Preprocessor:
    """Class to preprocess the input data."""

    def __init__(self, preprocessing_config: config.PreprocessingConfig) -> None:
        """Initialize the Preprocessor object.

        Args:
            preprocessing_config (config.PreprocessingConfig): The preprocessing configuration.
        """

        local_logger.info("Initializing Preprocessor with config: %s", preprocessing_config)

        self._mean = np.array(preprocessing_config.mean)
        self._std = np.array(preprocessing_config.std)

        self._spatial_config = preprocessing_config.spatial_config
        self._color_config = preprocessing_config.color_config
        self._resize_config = preprocessing_config.resize_config

        self._denormalizer = torchvision.transforms.Normalize(
            mean=-1 * self._mean / self._std,
            std=1 / self._std,
        )

        self._construct_transforms()

    def _construct_transforms(self) -> None:
        """Construct the data transforms."""

        self._normalization = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self._mean, self._std),
        ]

        self._spatial_augmentation = []
        if self._spatial_config:
            self._spatial_augmentation = get_spatial_transforms(
                width=self._resize_config.width,
                height=self._resize_config.height,
                hflip_prob=self._spatial_config.hflip_prob,
                vflip_prob=self._spatial_config.vflip_prob,
                max_rotate_in_degree=self._spatial_config.max_rotate_in_degree,
                allow_center_crop=self._spatial_config.allow_center_crop,
                allow_random_crop=self._spatial_config.allow_random_crop,
            )

        self._color_augmentation = []
        if self._color_config:
            self._color_augmentation = get_color_transforms(
                allow_gray_scale=self._color_config.allow_gray_scale,
                allow_random_color=self._color_config.allow_random_color,
            )

        self._resize_and_padding = get_resize_and_padding_transforms(
            width=self._resize_config.width,
            height=self._resize_config.height,
            interpolation=self._resize_config.interpolation,
            padding=self._resize_config.padding,
            maintain_aspect_ratio=self._resize_config.maintain_aspect_ratio,
        )

    def __call__(self, image: np.ndarray, is_augmented: bool) -> np.ndarray:
        """Preprocess the input image.

        Args:
            image (np.ndarray): The input image.
            is_augmented (bool): Whether the image is augmented or not.

        Returns:
            np.ndarray: The preprocessed image.
        """

        if is_augmented:
            layers = self._spatial_augmentation + self._color_augmentation
        else:
            layers = []
        layers += self._normalization + self._resize_and_padding

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

        return self._denormalizer(image)

    def get_training_transforms(self) -> torchvision.transforms.Compose:
        """Get the training transforms.

        Returns:
            torchvision.transforms.Compose: The training transforms.
        """

        return torchvision.transforms.Compose(
            self._spatial_augmentation + self._color_augmentation + self._normalization + self._resize_and_padding
        )

    def get_validation_transforms(self) -> torchvision.transforms.Compose:
        """Get the validation transforms.

        Returns:
            torchvision.transforms.Compose: The validation transforms.
        """

        return torchvision.transforms.Compose(self._normalization + self._resize_and_padding)

    def get_example_array(self) -> torch.Tensor:
        """Get an example array.

        Returns:
            torch.Tensor: The example array.
        """

        return torch.rand(1, 3, self._resize_config.height, self._resize_config.width)

    @staticmethod
    def compute_mean_and_std(dirpath: str) -> dict[str, list[float]]:
        """Compute the mean and standard deviation of the dataset.
        Suppose the mean and standard deviation are computed for each channel.
        So the output is expected to be a dictionary with the following format:
        {
            "mean": [mean_channel_1, mean_channel_2, mean_channel_3],
            "std": [std_channel_1, std_channel_2, std_channel_3],
        }

        Args:
            dirpath (str): The directory path.

        Returns:
            dict[str, list[float]]: The mean and standard deviation.
        """

        dataloader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                root=dirpath,
                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
            ),
            batch_size=1,
            shuffle=False,
        )
        mean, std = torch.zeros(3), torch.zeros(3)

        for images, _ in dataloader:
            mean += images.mean(dim=[0, 2, 3])
            std += images.std(dim=[0, 2, 3])

        mean /= len(dataloader)
        std /= len(dataloader)

        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
