import os

import pytest

from classifier_trains.schemas import config


@pytest.fixture(name="dataset_path")
def dataset_path_fixture() -> str:
    """Return the default directory path"""

    return f"{os.path.dirname(os.path.dirname(__file__))}/test_dataset"


def test_init_evaluation_config(dataset_path: str) -> None:
    """Test EvaluationConfig initialization"""

    c = config.EvaluationConfig(name="test", evalset_dir=dataset_path, models=[])
    assert c


def test_init_evaluation_config_with_invalid_evalset_dir() -> None:
    """Test EvaluationConfig initialization failed with invalid evalset_dir"""

    with pytest.raises(ValueError):
        config.EvaluationConfig(name="test", evalset_dir="INVALID", models=[])


def test_init_evaluation_config_with_invalid_model(dataset_path: str) -> None:
    """Test EvaluationConfig initialization failed with invalid model"""

    with pytest.raises(ValueError):
        config.EvaluationConfig(name="test", evalset_dir=dataset_path, models=["INVALID"])
