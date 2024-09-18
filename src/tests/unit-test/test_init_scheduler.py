import pytest
import torch

from classifier_trains.core.model.utils import initialize_scheduler
from classifier_trains.schemas.config import SchedulerConfig
from classifier_trains.schemas.constants import SchedulerType


@pytest.fixture(name="optimizer")
def optimizer_fixture() -> torch.optim.Optimizer:
    """Return a model parameter tensor."""

    model_params = torch.nn.Linear(2, 2).parameters()
    optimizer = torch.optim.SGD(model_params, lr=0.1)
    return optimizer


@pytest.fixture(name="num_epochs")
def num_epochs_fixture() -> int:
    """Return the number of epochs."""

    return 10


def test_initialize_scheduler_step(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
) -> None:
    """Test the initialize scheduler function with StepLR."""

    c = SchedulerConfig(name=SchedulerType.STEP, step_size=1, gamma=0.1)
    scheduler = initialize_scheduler(optimizer, c, num_epochs)
    assert scheduler.step_size == c.step_size
    assert scheduler.gamma == c.gamma
    assert scheduler.base_lrs == [0.1]


def test_initialize_scheduler_cosine(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
) -> None:
    """Test the initialize scheduler function with CosineAnnealingLR."""

    c = SchedulerConfig(name=SchedulerType.COSINE, lr_min=0.01)
    scheduler = initialize_scheduler(optimizer, c, num_epochs)
    assert scheduler.T_max == num_epochs
    assert scheduler.eta_min == c.lr_min
