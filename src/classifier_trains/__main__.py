import click

import classifier_trains.schemas.pipeline
from classifier_trains.core import ModelInterface
from classifier_trains.utils.file import load_yml


def _run_pipeline(pipeline_config: classifier_trains.schemas.pipeline.PipelineConfig, output_dir: str) -> None:
    """Run the pipeline based on the configuration file.

    Args:
        pipeline_config (classifier_trains.schemas.pipeline.PipelineConfig): Configuration for the pipeline.
        output_dir (str): Path to the output directory.
    """

    model_interface = ModelInterface(preprocessing_config=pipeline_config.preprocessing)

    if pipeline_config.enable_training:
        model_interface.train(
            model_config=pipeline_config.model,  # type: ignore
            training_config=pipeline_config.training,  # type: ignore
            dataloader_config=pipeline_config.dataloader,
            output_dir=output_dir,
        )

    if pipeline_config.enable_evaluation:
        model_interface.evaluate(
            evaluation_config=pipeline_config.evaluation,  # type: ignore
            dataloader_config=pipeline_config.dataloader,
            output_dir=output_dir,
        )


@click.group()
def cli() -> None:
    pass


@cli.command("run")
@click.option("--config-path", "-c", required=True, type=str, help="Path to the configuration file.")
@click.option("--output-dir", "-o", required=True, type=str, help="Path to the output directory.")
def run(config_path: str, output_dir: str) -> None:
    """Run Classifier Pipeline based on the configuration file.

    Args:
        config (str): Path to the configuration file.
        output_dir (str): Path to the output directory.

    Example:
        >>> python -m pipeline run -c config.yaml -o output
    """

    content = load_yml(config_path)
    pipeline_config = classifier_trains.schemas.pipeline.PipelineConfig(**content)
    _run_pipeline(pipeline_config=pipeline_config, output_dir=output_dir)


@cli.command("compute-mean-and-std")
@click.option("--dir-path", "-d", required=True, type=str, help="Path to the trainset directory.")
def compute_mean_and_std(dir_path: str) -> None:
    """Compute the mean and standard deviation of the dataset.

    Args:
        dir_path (str): Path to the trainset directory.

    Example:
        >>> python -m pipeline compute-mean-and-std -d dataset
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
        >>> python -m pipeline get-output-mapping -d dataset
    """

    output_mapping = ModelInterface.get_output_mapping(dirpath=dir_path)
    click.echo(output_mapping)


@cli.command("profile")
@click.option("--config-path", "-c", required=True, type=str, help="Path to the configuration file.")
@click.option("--output-dir", "-o", required=True, type=str, help="Path to the output directory.")
@click.option("--interval", "-i", default=0.001, type=float, help="Interval for the profiler.")
@click.option("--show-all", "-s", is_flag=True, help="Show all the functions in the profiler.")
@click.option("--timeline", "-t", is_flag=True, help="Show the timeline in the profiler.")
def profile(config_path: str, output_dir: str, interval: float, show_all: bool, timeline: bool) -> None:
    """Profile Classifier Pipeline based on the configuration file.

    Args:
        config_path (str): Path to the configuration file.
        output_dir (str): Path to the output directory.

    Example:
        >>> python -m pipeline profile -c config.yaml -o output
    """

    # Here we do not import the Profiler class at the top level
    # This is to allow the inexistence of dev dependencies
    try:
        from pyinstrument import Profiler  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError("Please install the dev-group package `pyinstrument` to use the profiler.") from exc

    profiler = Profiler(interval=interval)
    profiler.start()

    content = load_yml(config_path)
    pipeline_config = classifier_trains.schemas.pipeline.PipelineConfig(**content)
    _run_pipeline(pipeline_config=pipeline_config, output_dir=output_dir)

    profiler.stop()
    profiler.write_html(f"{output_dir}/profile.html", show_all=show_all, timeline=timeline)
    click.echo(profiler.output_text(unicode=True, color=True))


if __name__ == "__main__":
    cli()
