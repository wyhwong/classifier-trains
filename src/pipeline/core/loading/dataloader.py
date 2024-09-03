from typing import Optional

import lightning as pl
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from pipeline.schemas.config import DataloaderConfig
from pipeline.schemas.constants import Phase
from pipeline.utils import logger


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

        self.__trainset_dir = dataloader_config.trainset_dir
        self.__valset_dir = dataloader_config.valset_dir
        self.__test_dir = dataloader_config.testset_dir
        self.__batch_size = dataloader_config.batch_size
        self.__num_workers = dataloader_config.num_workers
        self.__transforms = transforms or {phase: None for phase in Phase}

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader."""

        return DataLoader(
            dataset=datasets.ImageFolder(
                root=self.__trainset_dir,
                transform=self.__transforms[Phase.TRAINING],
            ),
            batch_size=self.__batch_size,
            num_workers=self.__num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader."""

        return DataLoader(
            dataset=datasets.ImageFolder(
                root=self.__valset_dir,
                transform=self.__transforms[Phase.VALIDATION],
            ),
            batch_size=self.__batch_size,
            num_workers=self.__num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Get the test dataloader."""

        if not self.__test_dir:
            return None

        return DataLoader(
            dataset=datasets.ImageFolder(
                root=self.__test_dir,
                transform=self.__transforms[Phase.TESTING],
            ),
            batch_size=self.__batch_size,
            num_workers=self.__num_workers,
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

        transforms = self.__transforms[Phase.TRAINING] if is_augmented else self.__transforms[Phase.VALIDATION]
        dataloader = DataLoader(
            dataset=datasets.ImageFolder(
                root=dirpath,
                transform=transforms,
            ),
            batch_size=self.__batch_size,
            num_workers=self.__num_workers,
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
