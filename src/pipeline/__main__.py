import click
import yaml

from pipeline.core import ModelInterface
from pipeline.schemas import pipeline


@click.group()
def cli() -> None:
    pass


@cli.command("run")
@click.option("--config", "-c", required=True, type=str, help="Path to the configuration file.")
@click.option("--output-dir", "-o", required=True, type=str, help="Path to the output directory.")
def run(config: str, output_dir: str) -> None:
    """Run Classifier Pipeline based on the configuration file.

    Args:
        config (str): Path to the configuration file.
        output_dir (str): Path to the output directory.

    Example:
        >>> run("config.yaml", "output")
    """

    with open(config, mode="r", encoding="utf-8") as file:
        content = yaml.load(file, Loader=yaml.SafeLoader)

    pipeline_config = pipeline.PipelineConfig(**content)

    model_interface = ModelInterface(
        preprocessing_config=pipeline_config.preprocessing,
        model_config=pipeline_config.model,
    )

    if pipeline_config.enable_training:
        model_interface.train(
            training_config=pipeline_config.training,
            dataloader_config=pipeline_config.dataloader,
            output_dir=output_dir,
        )

    if pipeline_config.enable_evaluation:
        model_interface.evaluate(
            evaluation_config=pipeline_config.evaluation,
            output_dir=output_dir,
        )


@cli.command("compute-mean-and-std")
@click.option("--dir-path", "-d", required=True, type=str, help="Path to the trainset directory.")
def compute_mean_and_std(dir_path: str) -> None:
    """Compute the mean and standard deviation of the dataset.

    Args:
        dir_path (str): Path to the trainset directory.

    Example:
        >>> compute_mean_and_std("trainset")
    """

    mean_and_std = ModelInterface.compute_mean_and_std(dirpath=dir_path)
    click.echo(f"Mean: {mean_and_std['mean']}")
    click.echo(f"Std: {mean_and_std['std']}")


@cli.command("get-output-mapping")
@click.option("--dir-path", "-d", required=True, type=str, help="Path to the dataset directory.")
def get_output_mapping(dir_path: str) -> None:
    """Get the output mapping for the dataset.

    Args:
        dir_path (str): Path to the dataset directory.

    Example:
        >>> get_output_mapping("dataset")
    """

    output_mapping = ModelInterface.get_output_mapping(dirpath=dir_path)
    click.echo(output_mapping)


if __name__ == "__main__":
    cli()
