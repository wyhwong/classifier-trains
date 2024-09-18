import os

import pytest

from classifier_trains.schemas import config, constants, pipeline


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

    dirpath = f"{os.path.dirname(os.path.dirname(__file__))}/test_dataset"

    return config.DataloaderConfig(
        trainset_dir=dirpath,
        valset_dir=dirpath,
        batch_size=32,
        num_workers=4,
    )


@pytest.fixture(name="preprocessing")
def preprocessing_config_fixture() -> config.PreprocessingConfig:
    """PreprocessingConfig fixture"""

    return config.PreprocessingConfig(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        resize_config=config.ResizeConfig(
            width=224,
            height=224,
            interpolation=constants.InterpolationType.BILINEAR,
            padding=constants.PaddingType.CENTER,
            maintain_aspect_ratio=True,
        ),
        spatial_config=config.SpatialTransformConfig(
            hflip_prob=0.5,
            vflip_prob=0.5,
            max_rotate_in_degree=45,
            allow_center_crop=True,
            allow_random_crop=True,
        ),
        color_config=config.ColorTransformConfig(
            allow_gray_scale=True,
            allow_random_color=True,
        ),
    )


@pytest.fixture(name="training")
def training_config_fixture() -> config.TrainingConfig:
    """TrainingConfig fixture"""

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
    return c


@pytest.fixture(name="evaluation")
def evaluation_config_fixture() -> config.EvaluationConfig:
    """EvaluationConfig fixture"""

    c = config.EvaluationConfig(
        name="test",
        device="cpu",
        random_seed=42,
        precision=64,
        evalset_dir="test",
        models=[],
    )
    return c


def test_init_pipeline_config(
    model: config.ModelConfig,
    dataloader: config.DataloaderConfig,
    preprocessing: config.PreprocessingConfig,
    training: config.TrainingConfig,
    evaluation: config.EvaluationConfig,
):
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
):
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


def test_init_pipeline_config_without_evaluation(
    model: config.ModelConfig,
    dataloader: config.DataloaderConfig,
    preprocessing: config.PreprocessingConfig,
    training: config.TrainingConfig,
    evaluation: config.EvaluationConfig,
):
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
