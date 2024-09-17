import os

import pytest

from classifier_trains.utils.file import check_and_create_dir, load_yml, save_as_yml


@pytest.fixture(name="dirpath")
def default_dir_path() -> str:
    """Return the default directory path"""

    return f"{os.path.dirname(os.path.dirname(__file__))}/test_data"


def test_check_and_create_dir(dirpath: str) -> None:
    """Test check_and_create_dir function"""

    dirpath = f"{dirpath}/test_check_and_create_dir"
    check_and_create_dir(dirpath=dirpath)

    assert os.path.isdir(dirpath) is True
    os.removedirs(dirpath)


def test_load_yml(dirpath: str):
    """Test load_yml function"""

    filepath = f"{dirpath}/test_load_yml.yml"

    content = load_yml(filepath=filepath)
    assert content["as_expected"] is True


def test_save_as_yml(dirpath: str):
    """Test save_as_yml function"""

    content = {"as_expected": True}
    filepath = f"{dirpath}/test_save_as_yml.yml"
    save_as_yml(filepath=filepath, content=content)
    assert os.path.isfile(filepath) is True
    assert load_yml(filepath=filepath) == content
    os.remove(filepath)
