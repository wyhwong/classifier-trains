import click
import yaml

import pipeline.schemas.config


@click.command()
@click.option("--config", "-c", required=True, help="Path to the configuration file.")
def run(config: str) -> None:
    """Run Classifier Pipeline based on the configuration file."""

    with open(config, mode="r", encoding="utf-8") as file:
        content = yaml.load(file, Loader=yaml.SafeLoader)

    config = pipeline.schemas.config.PipelineConfig(**content)

    # TODO: Implement the pipeline based on the configuration file.
    print(config)


if __name__ == "__main__":
    run()
