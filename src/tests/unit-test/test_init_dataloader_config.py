import os

import pytest

from classifier_trains.schemas.config import DataloaderConfig


@pytest.fixture(name="dirpath")
def default_dir_path() -> str:
    """Return the default directory path"""

    return f"{os.path.dirname(os.path.dirname(__file__))}/test_dataset"


def test_init_dataloader_config(dirpath: str) -> None:
    """Test DataloaderConfig initialization"""

    c = DataloaderConfig(
        trainset_dir=dirpath,
        valset_dir=dirpath,
        batch_size=32,
        num_workers=4,
    )

    assert c


def test_init_dataloader_config_invalid_trainset_path(dirpath: str) -> None:
    """Test DataloaderConfig initialization with invalid directory path"""

    with pytest.raises(ValueError):
        DataloaderConfig(
            trainset_dir="invalid",
            valset_dir=dirpath,
            batch_size=32,
            num_workers=4,
        )


def test_init_dataloader_config_invalid_valset_path(dirpath: str) -> None:
    """Test DataloaderConfig initialization with invalid directory path"""

    with pytest.raises(ValueError):
        DataloaderConfig(
            trainset_dir=dirpath,
            valset_dir="invalid",
            batch_size=32,
            num_workers=4,
        )


def test_init_dataloader_config_invalid_testset_path(dirpath: str) -> None:
    """Test DataloaderConfig initialization with invalid directory path"""

    with pytest.raises(ValueError):
        DataloaderConfig(
            trainset_dir=dirpath,
            valset_dir=dirpath,
            batch_size=32,
            num_workers=4,
            testset_dir="invalid",
        )
