import pytest
import torch

from classifier_trains.core.model.utils import initialize_optimizer
from classifier_trains.schemas.config import OptimizerConfig
from classifier_trains.schemas.constants import OptimizerType


@pytest.fixture(name="model_params")
def model_params_fixture() -> torch.Tensor:
    """Return a model parameter tensor."""

    return torch.nn.Linear(2, 2).parameters()


def optimizer_config_factory(name: OptimizerType) -> OptimizerConfig:
    """Return an optimizer configuration object."""

    return OptimizerConfig(name=name, lr=0.07, weight_decay=0.004)


def test_initialize_optimizer_sgd(model_params: list[float]) -> None:
    """Test the initialize optimizer function with SGD."""

    optimizer_config = optimizer_config_factory(OptimizerType.SGD)
    optimizer = initialize_optimizer(model_params, optimizer_config)
    assert optimizer.defaults["lr"] == optimizer_config.lr
    assert optimizer.defaults["weight_decay"] == optimizer_config.weight_decay


def test_initialize_optimizer_rmsprop(model_params: list[float]) -> None:
    """Test the initialize optimizer function with RMSprop."""

    optimizer_config = optimizer_config_factory(OptimizerType.RMSPROP)
    optimizer = initialize_optimizer(model_params, optimizer_config)
    assert optimizer.defaults["lr"] == optimizer_config.lr
    assert optimizer.defaults["weight_decay"] == optimizer_config.weight_decay


def test_initialize_optimizer_adam(model_params: list[float]) -> None:
    """Test the initialize optimizer function with Adam."""

    optimizer_config = optimizer_config_factory(OptimizerType.ADAM)
    optimizer = initialize_optimizer(model_params, optimizer_config)
    assert optimizer.defaults["lr"] == optimizer_config.lr
    assert optimizer.defaults["weight_decay"] == optimizer_config.weight_decay


def test_initialize_optimizer_adamw(model_params: list[float]) -> None:
    """Test the initialize optimizer function with AdamW."""

    optimizer_config = optimizer_config_factory(OptimizerType.ADAMW)
    optimizer = initialize_optimizer(model_params, optimizer_config)
    assert optimizer.defaults["lr"] == optimizer_config.lr
    assert optimizer.defaults["weight_decay"] == optimizer_config.weight_decay
