from typing import Optional

import lightning as pl
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from classifier_trains.schemas.config import DataloaderConfig
from classifier_trains.schemas.constants import Phase
from classifier_trains.utils import logger


local_logger = logger.get_logger(__name__)


class ImageDataloader(pl.LightningDataModule):
    """Class to handle the image dataloader."""

    def __init__(
        self,
        dataloader_config: DataloaderConfig,
        transforms: Optional[dict[Phase, torchvision.transforms.Compose]] = None,
    ) -> None:
        """Initialize the ImageDataloader object.

        Args:
            dataloader_config (DataloaderConfig): The dataloader configuration.
            transforms (Optional[dict[Phase, torchvision.transforms.Compose]]): The transforms for each phase.
        """

        super().__init__()

        local_logger.info("Initializing ImageDataloader with config: %s", dataloader_config)

        self._batch_size = dataloader_config.batch_size
        self._num_workers = dataloader_config.num_workers
        self._transforms = transforms or {phase: None for phase in Phase}

        self._trainset_dir: Optional[str] = None
        self._valset_dir: Optional[str] = None
        self._test_dir: Optional[str] = None

    def setup_for_training(self, trainset_dir: str, valset_dir: str, test_dir: Optional[str] = None) -> None:
        """Setup the dataloader.

        Args:
            trainset_dir (str): The path to the training dataset.
            valset_dir (str): The path to the validation dataset.
            test_dir (Optional[str]): The path to the test dataset.
        """

        local_logger.info(
            "Setting up dataloader with trainset_dir: %s, valset_dir: %s, test_dir: %s",
            trainset_dir,
            valset_dir,
            test_dir,
        )

        self._trainset_dir = trainset_dir
        self._valset_dir = valset_dir
        self._test_dir = test_dir

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader."""

        if not self._trainset_dir:
            raise ValueError("trainset_dir is not set.")

        return DataLoader(
            dataset=datasets.ImageFolder(
                root=self._trainset_dir,
                transform=self._transforms[Phase.TRAINING],
            ),
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader."""

        if not self._valset_dir:
            raise ValueError("valset_dir is not set.")

        return DataLoader(
            dataset=datasets.ImageFolder(
                root=self._valset_dir,
                transform=self._transforms[Phase.VALIDATION],
            ),
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Get the test dataloader."""

        if not self._test_dir:
            local_logger.info("testset_dir is not set. Return None.")
            return None

        return DataLoader(
            dataset=datasets.ImageFolder(
                root=self._test_dir,
                transform=self._transforms[Phase.TESTING],
            ),
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
        )

    def get_dataloader(self, dirpath: str, is_augmented=False) -> DataLoader:
        """Get the dataloader for the dataset.

        Args:
            dirpath (str): The path to the dataset.
            is_augmented (bool): If the dataset is augmented.

        Returns:
            DataLoader: The dataloader.
        """

        transforms = self._transforms[Phase.TRAINING] if is_augmented else self._transforms[Phase.VALIDATION]
        dataloader = DataLoader(
            dataset=datasets.ImageFolder(
                root=dirpath,
                transform=transforms,
            ),
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=False,
        )
        return dataloader

    @staticmethod
    def get_output_mapping(dirpath: str) -> dict[str, int]:
        """Get the output mapping for the dataset.

        Args:
            dirpath (str): The path to the dataset.

        Returns:
            dict[str, int]: The output mapping.
        """

        dataset = datasets.ImageFolder(root=dirpath)
        mapping = dataset.class_to_idx

        local_logger.info("Output mapping: %s", mapping)
        return mapping
