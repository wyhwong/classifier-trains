import click
import yaml

from pipeline.core import ModelInterface
import pipeline.schemas.config


@click.command()
@click.option("--config", "-c", required=True, type=str, help="Path to the configuration file.")
def run(config: str) -> None:
    """Run Classifier Pipeline based on the configuration file."""

    with open(config, mode="r", encoding="utf-8") as file:
        content = yaml.load(file, Loader=yaml.SafeLoader)

    config = pipeline.schemas.config.PipelineConfig(**content)

    model_interface = ModelInterface(
        preprocessing_config=config.preprocessing,
        model_config=config.model,
    )

    # TODO: Implement the pipeline based on the configuration file.


if __name__ == "__main__":
    run()
