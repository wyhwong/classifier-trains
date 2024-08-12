from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

import pipeline.core.utils
import pipeline.logger
from pipeline.schemas.visualization import Labels, Line, Padding


local_logger = pipeline.logger.get_logger(__name__)


def initialize_plot(nrows=1, ncols=1, figsize=(10, 6), labels=Labels(), padding=Padding(), fontsize=12):
    """
    Initialize a plot from matplotlib.

    Args:
    -----
        nrows (int):
            The number of rows of the plot.

        ncols (int):
            The number of columns of the plot.

        figsize (tuple):
            The size of the plot.

        labels (Labels):
            The labels of the plot.

        padding (Padding):
            The padding of the plot.

        fontsize (int):
            The fontsize of the plot.

    Returns:
    -----
        fig (matplotlib.figure.Figure):
            The figure of the plot.

        axes (numpy.ndarray):
            The axes of the plot.
    """

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    local_logger.debug("Initialized plot: nrows=%d, ncols=%d, figsize=%s.", nrows, ncols, figsize)

    fig.tight_layout(pad=padding.tpad)
    fig.subplots_adjust(left=padding.lpad, bottom=padding.bpad)
    local_logger.debug(
        "Adjusted plot: tpad=%f, lpad=%f, bpad=%f.",
        padding.tpad,
        padding.lpad,
        padding.bpad,
    )

    fig.suptitle(labels.title, fontsize=fontsize)
    fig.text(
        x=0.04,
        y=0.5,
        s=labels.ylabel,
        fontsize=fontsize,
        rotation="vertical",
        verticalalignment="center",
    )
    fig.text(x=0.5, y=0.04, s=labels.xlabel, fontsize=fontsize, horizontalalignment="center")
    local_logger.debug(
        "Added title and labels: title=%s, xlabel=%s, ylabel=%s.",
        labels.title,
        labels.xlabel,
        labels.ylabel,
    )

    return (fig, axes)


def add_lines_to_plot(ax, lines: list[Line]) -> None:
    """
    Add lines to the plot.

    Args:
    -----
        ax (matplotlib.axes.Axes):
            The axes of the plot.

        lines (list[Line]):
            The lines to be added to the plot.

    Returns:
    -----
        None
    """

    for line in lines:
        if not isinstance(line, Line):
            local_logger.error("Expected Line, got %s.", type(line))
            raise TypeError(f"Expected Line, got {type(line)}.")

        sns.lineplot(
            x=[line.left_bottom[0], line.right_top[0]],
            y=[line.left_bottom[1], line.right_top[1]],
            color=line.color,
            linestyle=line.linestyle,
            label=line.label,
            ax=ax,
        )


def savefig_and_close(filename: Optional[str] = None, output_dir: Optional[str] = None, close=True) -> None:
    """
    Save the figure and close it.

    Args:
    -----
        filename (str):
            The filename of the figure.

        output_dir (str):
            The output directory of the figure.

        close (bool):
            Whether to close the figure.

    Returns:
    -----
        None
    """

    if output_dir:
        pipeline.core.utils.check_and_create_dir(output_dir)
        savepath = f"{output_dir}/{filename}"
        plt.savefig(savepath, facecolor="w")
        local_logger.info("Saved figure to %s.", savepath)

    if close:
        plt.close()
        local_logger.info("Closed figure.")
