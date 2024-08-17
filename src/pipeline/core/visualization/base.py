from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

import pipeline.core.utils
import pipeline.logger
from pipeline.schemas.visualization import Labels, Line, Padding


local_logger = pipeline.logger.get_logger(__name__)


def initialize_plot(nrows=1, ncols=1, figsize=(10, 6), labels=Labels(), padding=Padding(), fontsize=12):
    """Initialize a plot from matplotlib.

    Args:
        nrows (int, optional): Number of rows. Defaults to 1.
        ncols (int, optional): Number of columns. Defaults to 1.
        figsize (tuple, optional): Figure size. Defaults to (10, 6).
        labels (Labels, optional): Labels for the plot. Defaults to Labels().
        padding (Padding, optional): Padding for the plot. Defaults to Padding().
        fontsize (int, optional): Font size. Defaults to 12.

    Returns:
        tuple: Figure and axes.
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
    """Add lines to the plot.

    Args:
        ax (Axes): The axes of the plot.
        lines (list[Line]): The lines to add to the plot.

    Raises:
        TypeError: If the line is not an instance of Line
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
        filename (str, optional): The filename. Defaults to None.
        output_dir (str, optional): The output directory. Defaults to None.
        close (bool, optional): Whether to close the figure. Defaults to True.

    Raises:
        FileNotFoundError: If the output directory does not exist.
    """

    if output_dir:
        pipeline.core.utils.check_and_create_dir(output_dir)
        savepath = f"{output_dir}/{filename}"
        plt.savefig(savepath, facecolor="w")
        local_logger.info("Saved figure to %s.", savepath)

    if close:
        plt.close()
        local_logger.info("Closed figure.")
