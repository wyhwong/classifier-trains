import os

import pytest

from classifier_trains.core import ModelInterface
from classifier_trains.schemas.pipeline import PipelineConfig
from classifier_trains.utils.file import load_yml


@pytest.fixture(name="config")
def config_path_fixture() -> PipelineConfig:
    """Return the default configuration path"""

    config_path = f"{os.path.dirname(os.path.dirname(__file__))}/test_data/pipeline_config_only_train.yml"
    content = load_yml(config_path)
    return PipelineConfig(**content)


def test_training(config: PipelineConfig) -> None:
    """Test the training function"""

    model_interface = ModelInterface(preprocessing_config=config.preprocessing)

    model_interface.train(
        model_config=config.model,
        training_config=config.training,
        dataloader_config=config.dataloader,
        output_dir="output",
    )

    # Check if the output directory is created
    dirpath = f"output/train-{config.training.name}"
    assert os.path.exists(dirpath)
    # Suppose we have folder inside the output directory indicating the version
    # In that folder, we should have:
    # - best.ckpt
    # - last.ckpt
    # - training.yml
    # - hparams.yaml
    # - a file with tensorboard logs (with random name)
    version = os.listdir(dirpath)[0]
    assert len(os.listdir(f"{dirpath}/{version}")) == 5
    assert os.path.exists(f"{dirpath}/{version}/training.yml")
    assert os.path.exists(f"{dirpath}/{version}/hparams.yaml")
    assert os.path.exists(f"{dirpath}/{version}/best.ckpt")
    assert os.path.exists(f"{dirpath}/{version}/last.ckpt")

    # Clean up the output directory
    os.system("rm -rf output")
    assert not os.path.exists("output")
