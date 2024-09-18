import os

import pytest

from classifier_trains.schemas import config, constants, pipeline


@pytest.fixture(name="dataset_path")
def dataset_path_fixture() -> str:
    """Return the default directory path"""

    return f"{os.path.dirname(os.path.dirname(__file__))}/test_dataset"


@pytest.fixture(name="model")
def model_config_fixture() -> config.ModelConfig:
    """ModelConfig fixture"""

    return config.ModelConfig(
        backbone=constants.ModelBackbone.RESNET50,
        num_classes=10,
        weights="DEFAULT",
    )


@pytest.fixture(name="dataloader")
def dataloader_config_fixture() -> config.DataloaderConfig:
    """DataloaderConfig fixture"""

    return config.DataloaderConfig(batch_size=32)


@pytest.fixture(name="preprocessing")
def preprocessing_config_fixture() -> config.PreprocessingConfig:
    """PreprocessingConfig fixture"""

    return config.PreprocessingConfig(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        resize_config=config.ResizeConfig(width=48, height=48),
    )


@pytest.fixture(name="optimizer")
def optimizer_config_fixture() -> config.OptimizerConfig:
    """OptimizerConfig fixture"""

    return config.OptimizerConfig(name=constants.OptimizerType.ADAM, lr=1e-3, weight_decay=1e-4)


@pytest.fixture(name="scheduler")
def scheduler_config_fixture() -> config.SchedulerConfig:
    """SchedulerConfig fixture"""

    return config.SchedulerConfig(name=constants.SchedulerType.STEP, step_size=1, gamma=0.1)


@pytest.fixture(name="training")
def training_config_fixture(
    optimizer: config.OptimizerConfig,
    scheduler: config.SchedulerConfig,
    dataset_path: str,
) -> config.TrainingConfig:
    """TrainingConfig fixture"""

    return config.TrainingConfig(
        name="test",
        num_epochs=10,
        optimizer=optimizer,
        scheduler=scheduler,
        trainset_dir=dataset_path,
        valset_dir=dataset_path,
    )


@pytest.fixture(name="evaluation")
def evaluation_config_fixture(dataset_path: str) -> config.EvaluationConfig:
    """EvaluationConfig fixture"""

    return config.EvaluationConfig(name="test", evalset_dir=dataset_path, models=[])


def test_init_pipeline_config(
    model: config.ModelConfig,
    dataloader: config.DataloaderConfig,
    preprocessing: config.PreprocessingConfig,
    training: config.TrainingConfig,
    evaluation: config.EvaluationConfig,
) -> None:
    """Test PipelineConfig initialization"""

    c = pipeline.PipelineConfig(
        enable_training=True,
        enable_evaluation=True,
        model=model,
        dataloader=dataloader,
        preprocessing=preprocessing,
        training=training,
        evaluation=evaluation,
    )

    assert c


def test_init_pipeline_config_without_training(
    model: config.ModelConfig,
    dataloader: config.DataloaderConfig,
    preprocessing: config.PreprocessingConfig,
    training: config.TrainingConfig,
    evaluation: config.EvaluationConfig,
) -> None:
    """Test PipelineConfig initialization"""

    c = pipeline.PipelineConfig(
        enable_training=False,
        enable_evaluation=True,
        model=model,
        dataloader=dataloader,
        preprocessing=preprocessing,
        training=training,
        evaluation=evaluation,
    )

    # Expected: training config is ignored
    assert c.training is None
    # Expected: model config is ignored
    assert c.model is None


def test_init_pipeline_config_with_training_but_empty_content(
    model: config.ModelConfig,
    dataloader: config.DataloaderConfig,
    preprocessing: config.PreprocessingConfig,
    evaluation: config.EvaluationConfig,
) -> None:
    """Test PipelineConfig initialization"""

    with pytest.raises(ValueError):
        pipeline.PipelineConfig(
            enable_training=True,
            enable_evaluation=True,
            model=model,
            dataloader=dataloader,
            preprocessing=preprocessing,
            training=None,
            evaluation=evaluation,
        )


def test_init_pipeline_config_with_trainig_but_no_model(
    dataloader: config.DataloaderConfig,
    preprocessing: config.PreprocessingConfig,
    training: config.TrainingConfig,
    evaluation: config.EvaluationConfig,
) -> None:
    """Test PipelineConfig initialization"""

    with pytest.raises(ValueError):
        pipeline.PipelineConfig(
            enable_training=True,
            enable_evaluation=True,
            model=None,
            dataloader=dataloader,
            preprocessing=preprocessing,
            training=training,
            evaluation=evaluation,
        )


def test_init_pipeline_config_without_evaluation(
    model: config.ModelConfig,
    dataloader: config.DataloaderConfig,
    preprocessing: config.PreprocessingConfig,
    training: config.TrainingConfig,
    evaluation: config.EvaluationConfig,
) -> None:
    """Test PipelineConfig initialization"""

    c = pipeline.PipelineConfig(
        enable_training=True,
        enable_evaluation=False,
        model=model,
        dataloader=dataloader,
        preprocessing=preprocessing,
        training=training,
        evaluation=evaluation,
    )

    # Expected: evaluation config is ignored
    assert c.evaluation is None


def test_init_pipeline_config_with_evaluation_but_empty_content(
    model: config.ModelConfig,
    dataloader: config.DataloaderConfig,
    preprocessing: config.PreprocessingConfig,
    training: config.TrainingConfig,
) -> None:
    """Test PipelineConfig initialization"""

    with pytest.raises(ValueError):
        pipeline.PipelineConfig(
            enable_training=True,
            enable_evaluation=True,
            model=model,
            dataloader=dataloader,
            preprocessing=preprocessing,
            training=training,
            evaluation=None,
        )


def test_init_pipeline_config_inconsistent_random_seed(
    model: config.ModelConfig,
    dataloader: config.DataloaderConfig,
    preprocessing: config.PreprocessingConfig,
    training: config.TrainingConfig,
    evaluation: config.EvaluationConfig,
) -> None:
    """Test PipelineConfig initialization"""

    training.random_seed = 42
    evaluation.random_seed = 43

    with pytest.raises(ValueError):
        pipeline.PipelineConfig(
            enable_training=True,
            enable_evaluation=True,
            model=model,
            dataloader=dataloader,
            preprocessing=preprocessing,
            training=training,
            evaluation=evaluation,
        )
