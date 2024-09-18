import os

import pytest

from classifier_trains.core import ModelInterface


@pytest.fixture(name="dataset_path")
def dataset_path_fixture() -> str:
    """Return the default directory path"""

    return f"{os.path.dirname(os.path.dirname(__file__))}/test_dataset"


def test_get_output_mapping(dataset_path: str) -> None:
    """Test the get output mapping function"""

    output_mapping = ModelInterface.get_output_mapping(dirpath=dataset_path)

    assert output_mapping["cats"] == 0
    assert output_mapping["dogs"] == 1
