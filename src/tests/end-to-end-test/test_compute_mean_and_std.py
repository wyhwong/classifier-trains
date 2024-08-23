import os

import numpy as np
import pytest

from pipeline.core import ModelInterface


@pytest.fixture(name="dirpath")
def default_dir_path() -> str:
    """Return the default directory path"""

    return f"{os.path.dirname(os.path.dirname(__file__))}/test_dataset"


def test_compute_mean_and_std(dirpath: str) -> None:
    """Test the compute mean and std function"""

    mean_and_std = ModelInterface.compute_mean_and_std(dirpath=dirpath)

    assert np.isclose(mean_and_std["mean"][0], 0.47488933801651, atol=1e-9)
    assert np.isclose(mean_and_std["mean"][1], 0.4632834792137146, atol=1e-9)
    assert np.isclose(mean_and_std["mean"][2], 0.4371258318424225, atol=1e-9)

    assert np.isclose(mean_and_std["std"][0], 0.21972933411598206, atol=1e-9)
    assert np.isclose(mean_and_std["std"][1], 0.22210058569908142, atol=1e-9)
    assert np.isclose(mean_and_std["std"][2], 0.22161932289600372, atol=1e-9)
