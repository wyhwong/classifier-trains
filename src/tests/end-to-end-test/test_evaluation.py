import os

import pytest

from classifier_trains.core import ModelInterface
from classifier_trains.schemas.pipeline import PipelineConfig
from classifier_trains.utils.file import load_yml


@pytest.fixture(name="config")
def config_path_fixture() -> PipelineConfig:
    """Return the default configuration path"""

    config_path = f"{os.path.dirname(os.path.dirname(__file__))}/test_data/pipeline_config_only_eval.yml"
    content = load_yml(config_path)
    return PipelineConfig(**content)


def test_evaluation(config: PipelineConfig) -> None:
    """Test the evaluation function"""

    model_interface = ModelInterface(preprocessing_config=config.preprocessing)

    model_interface.evaluate(
        evaluation_config=config.evaluation,
        dataloader_config=config.dataloader,
        output_dir="output",
    )

    # Check if the output directory is created
    assert os.path.exists(f"output/eval-{config.evaluation.name}")
    # Suppose we have folder inside the output directory indicating the version
    # In that folder, we should have:
    # - evaluation.yml
    # - hparams.yml
    # - a file with tensorboard logs (with random name)
    version = os.listdir(f"output/eval-{config.evaluation.name}")[0]
    assert len(os.listdir(f"output/eval-{config.evaluation.name}/{version}")) == 3
    assert os.path.exists(f"output/eval-{config.evaluation.name}/{version}/evaluation.yml")
    assert os.path.exists(f"output/eval-{config.evaluation.name}/{version}/hparams.yaml")

    # Clean up the output directory
    os.system("rm -rf output")
    assert not os.path.exists("output")
