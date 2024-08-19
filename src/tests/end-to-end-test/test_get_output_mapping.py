import os

import pytest

from pipeline.core import ModelInterface


@pytest.fixture(name="dirpath")
def default_dir_path() -> str:
    """Return the default directory path"""

    return f"{os.path.dirname(os.path.dirname(__file__))}/test_dataset"


def test_get_output_mapping(dirpath: str) -> None:
    """Test the get output mapping function"""

    output_mapping = ModelInterface.get_output_mapping(dirpath=dirpath)

    assert output_mapping["cats"] == 0
    assert output_mapping["dogs"] == 1
