import click
import yaml

from pipeline.core import ModelInterface
from pipeline.schemas import pipeline


@click.command()
@click.option("--config", "-c", required=True, type=str, help="Path to the configuration file.")
@click.option("--output-dir", "-o", required=True, type=str, help="Path to the output directory.")
def run(config: str, output_dir: str) -> None:
    """Run Classifier Pipeline based on the configuration file."""

    with open(config, mode="r", encoding="utf-8") as file:
        content = yaml.load(file, Loader=yaml.SafeLoader)

    config = pipeline.PipelineConfig(**content)

    model_interface = ModelInterface(
        preprocessing_config=config.preprocessing,
        model_config=config.model,
    )

    model_interface.train(
        training_config=config.training,
        dataloader_config=config.dataloader,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    run()
