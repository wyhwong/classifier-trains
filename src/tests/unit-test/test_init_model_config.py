from pathlib import Path

import pytest

from classifier_trains.schemas import config, constants


def test_init_model_config():
    """Test ModelConfig initialization"""

    for backbone in constants.ModelBackbone:
        c = config.ModelConfig(
            backbone=backbone,
            num_classes=10,
            weights="DEFAULT",
        )

        assert c


def test_init_model_config_with_checkpoint_path():
    """Test ModelConfig initialization with checkpoint path"""

    Path("checkpoint.ckpt").touch()
    c = config.ModelConfig(
        backbone=constants.ModelBackbone.RESNET18,
        num_classes=10,
        weights="DEFAULT",
        checkpoint_path="checkpoint.ckpt",
    )
    assert c

    Path("checkpoint.ckpt").unlink()


def test_init_model_config_failed_with_invalid_checkpoint_path():
    """Test ModelConfig initialization failed with invalid checkpoint path"""

    with pytest.raises(ValueError):
        config.ModelConfig(
            backbone=constants.ModelBackbone.RESNET18,
            num_classes=10,
            weights="DEFAULT",
            checkpoint_path="INVALID",
        )


def test_init_model_config_failed_with_unexisting_checkpoint_path():
    """Test ModelConfig initialization failed with unexisting checkpoint path"""

    with pytest.raises(ValueError):
        config.ModelConfig(
            backbone=constants.ModelBackbone.RESNET18,
            num_classes=10,
            weights="DEFAULT",
            checkpoint_path="unexisting.ckpt",
        )


def test_init_model_config_failed_with_invalid_backbone():
    """Test ModelConfig initialization failed with invalid backbone"""

    with pytest.raises(ValueError):
        config.ModelConfig(
            backbone="INVALID",
            num_classes=10,
            weights="DEFAULT",
        )


def test_init_model_config_failed_with_invalid_num_classes():
    """Test ModelConfig initialization failed with invalid num_classes"""

    with pytest.raises(ValueError):
        config.ModelConfig(
            backbone=constants.ModelBackbone.RESNET18,
            num_classes=-1,
            weights="DEFAULT",
        )
