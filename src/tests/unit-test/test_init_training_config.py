import pytest

from classifier_trains.schemas import config, constants


def test_init_optimizer_config():
    """Test OptimizerConfig initialization"""

    for optimizer in constants.OptimizerType:
        c = config.OptimizerConfig(name=optimizer, lr=1e-3, weight_decay=1e-4)
        assert c


def test_init_optimizer_config_failed_with_invalid_optimizer():
    """Test OptimizerConfig initialization failed with invalid optimizer"""

    with pytest.raises(ValueError):
        config.OptimizerConfig(name="INVALID", lr=1e-3, weight_decay=1e-4)


def test_init_optimizer_config_failed_with_invalid_lr():
    """Test OptimizerConfig initialization failed with invalid lr"""

    with pytest.raises(ValueError):
        config.OptimizerConfig(name=constants.OptimizerType.ADAM, lr=-1, weight_decay=1e-4)

    with pytest.raises(ValueError):
        config.OptimizerConfig(name=constants.OptimizerType.ADAM, lr=0, weight_decay=1e-4)


def test_init_optimizer_config_failed_with_invalid_weight_decay():
    """Test OptimizerConfig initialization failed with invalid weight_decay"""

    with pytest.raises(ValueError):
        config.OptimizerConfig(name=constants.OptimizerType.ADAM, lr=1e-3, weight_decay=-1)


def test_init_optimizer_config_with_extra_params_for_adam():
    """Test OptimizerConfig initialization with extra params for Adam"""

    with pytest.raises(ValueError):
        # Extra parameter alpha
        config.OptimizerConfig(
            name=constants.OptimizerType.ADAM,
            lr=1e-3,
            weight_decay=1e-4,
            alpha=0.9,
        )

    with pytest.raises(ValueError):
        # Extra parameter momentum
        config.OptimizerConfig(
            name=constants.OptimizerType.ADAM,
            lr=1e-3,
            weight_decay=1e-4,
            momentum=0.9,
        )


def test_init_optimizer_config_with_extra_params_for_adamw():
    """Test OptimizerConfig initialization with extra params for AdamW"""

    with pytest.raises(ValueError):
        # Extra parameter alpha
        config.OptimizerConfig(
            name=constants.OptimizerType.ADAMW,
            lr=1e-3,
            weight_decay=1e-4,
            alpha=0.9,
        )

    with pytest.raises(ValueError):
        # Extra parameter momentum
        config.OptimizerConfig(
            name=constants.OptimizerType.ADAMW,
            lr=1e-3,
            weight_decay=1e-4,
            momentum=0.9,
        )


def test_init_optimizer_config_with_extra_params_for_sgd():
    """Test OptimizerConfig initialization with extra params for SGD"""

    with pytest.raises(ValueError):
        # Extra parameter alpha
        config.OptimizerConfig(
            name=constants.OptimizerType.SGD,
            lr=1e-3,
            weight_decay=1e-4,
            alpha=0.9,
        )

    with pytest.raises(ValueError):
        # Extra parameter betas
        config.OptimizerConfig(
            name=constants.OptimizerType.SGD,
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )


def test_init_optimizer_config_with_extra_params_for_rmsprop():
    """Test OptimizerConfig initialization with extra params for RMSprop"""

    with pytest.raises(ValueError):
        # Extra parameter betas
        config.OptimizerConfig(
            name=constants.OptimizerType.RMSPROP,
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )


def test_init_steplr_scheduler_config():
    """Test StepLR SchedulerConfig initialization"""

    c = config.SchedulerConfig(name=constants.SchedulerType.STEP, step_size=1, gamma=0.1)

    assert c


def test_init_steplr_scheduler_config_failed_with_invalid_step_size():
    """Test StepLR SchedulerConfig initialization failed with invalid step_size"""

    with pytest.raises(ValueError):
        config.SchedulerConfig(name=constants.SchedulerType.STEP, step_size=-1, gamma=0.1)


def test_init_steplr_scheduler_config_failed_with_invalid_gamma():
    """Test StepLR SchedulerConfig initialization failed with invalid gamma"""

    with pytest.raises(ValueError):
        config.SchedulerConfig(name=constants.SchedulerType.STEP, step_size=1, gamma=-1)


def test_init_steplr_scheduler_config_failed_with_extra_params():
    """Test StepLR SchedulerConfig initialization failed with extra params"""

    with pytest.raises(ValueError):
        config.SchedulerConfig(name=constants.SchedulerType.STEP, step_size=1, gamma=0.1, lr_min=0.01)


def test_init_cosine_scheduler_config():
    """Test CosineAnnealingLR SchedulerConfig initialization"""

    c = config.SchedulerConfig(name=constants.SchedulerType.COSINE, lr_min=0.01)

    assert c


def test_init_cosine_scheduler_config_failed_with_invalid_lr_min():
    """Test CosineAnnealingLR SchedulerConfig initialization failed with invalid lr_min"""

    with pytest.raises(ValueError):
        config.SchedulerConfig(name=constants.SchedulerType.COSINE, lr_min=-1)


def test_init_cosine_scheduler_config_failed_with_extra_params():
    """Test CosineAnnealingLR SchedulerConfig initialization failed with extra params"""

    with pytest.raises(ValueError):
        # Extra parameter gamma
        config.SchedulerConfig(name=constants.SchedulerType.COSINE, lr_min=0.01, step_size=1)

    with pytest.raises(ValueError):
        # Extra parameter step_size
        config.SchedulerConfig(name=constants.SchedulerType.COSINE, lr_min=0.01, gamma=0.1)


def test_init_training_config():
    """Test TrainingConfig initialization"""

    optimizer = config.OptimizerConfig(name=constants.OptimizerType.ADAM, lr=1e-3, weight_decay=1e-4)
    scheduler = config.SchedulerConfig(name=constants.SchedulerType.STEP, step_size=1, gamma=0.1)

    c = config.TrainingConfig(
        name="test",
        num_epochs=10,
        random_seed=42,
        precision=64,
        device="cpu",
        validate_every_n_epoch=1,
        save_every_n_epoch=1,
        patience_in_epoch=1,
        criterion=constants.Criterion.LOSS,
        optimizer=optimizer,
        scheduler=scheduler,
        export_best_as_onnx=True,
        export_last_as_onnx=True,
        max_num_hrs=1.0,
    )

    assert c
