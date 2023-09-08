import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import dataclasses

import utils.common
import utils.logger
import utils.preprocessing

LOGGER = utils.logger.get_logger("utils/visualization")


@dataclasses.dataclass
class Padding:
    """
    Padding for the plot.

    Attributes
    ----------
    tpad: float, the top padding of the plot.
    lpad: float, the left padding of the plot.
    bpad: float, the bottom padding of the plot.
    """

    tpad: float = 2.5
    lpad: float = 0.1
    bpad: float = 0.12


@dataclasses.dataclass
class Labels:
    """
    Labels for the plot.

    Attributes
    ----------
    title: str, the title of the plot.
    xlabel: str, the x-axis label of the plot.
    ylabel: str, the y-axis label of the plot.
    zlabel: str, the z-axis label of the plot.
    """

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""


def initialize_plot(
    nrows=1, ncols=1, figsize=(10, 6), labels=Labels(), padding=Padding(), fontsize=12
) -> tuple:
    """
    Initialize the plot with the given nrows, ncols, figsize, labels, padding, and fontsize.

    Parameters
    ----------
    nrows: int, optional, the number of rows of the plot.
    ncols: int, optional, the number of columns of the plot.
    figsize: tuple, optional, the figure size of the plot.
    labels: Labels, optional, the labels of the plot.
    padding: Padding, optional, the padding of the plot.
    fontsize: int, optional, the fontsize of the plot.

    Returns
    -------
    fig: matplotlib.figure.Figure, the figure of the plot.
    axes: matplotlib.axes.Axes, the axes of the plot.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.tight_layout(pad=padding.tpad)
    fig.subplots_adjust(left=padding.lpad, bottom=padding.bpad)
    fig.suptitle(labels.title, fontsize=fontsize)
    fig.text(
        x=0.04,
        y=0.5,
        s=labels.ylabel,
        fontsize=fontsize,
        rotation="vertical",
        verticalalignment="center",
    )
    fig.text(
        x=0.5, y=0.04, s=labels.xlabel, fontsize=fontsize, horizontalalignment="center"
    )
    return (fig, axes)


def savefig_and_close(filename: str, output_dir=None, close=True) -> None:
    """
    Save the plot as an image and close the plot.

    Parameters
    ----------
    filename: str, required, the filename of the plot.
    output_dir: str, optional, the output directory of the plot.
    close: bool, optional, whether to close the plot or not.

    Returns
    -------
    None
    """
    if output_dir:
        utils.common.check_and_create_dir(output_dir)
        savepath = f"{output_dir}/{filename}"
        plt.savefig(savepath, facecolor="w")
        LOGGER.info(f"Saved plot at {savepath}.")
    if close:
        plt.close()


def visualize_acc_and_loss(
    train_loss: dict[str, list], train_acc: dict[str, list], output_dir=None, close=True
) -> None:
    """
    Visualize the training/validation loss and accuracy during training.

    Parameters
    ----------
    train_loss: dict, required, the training/validation loss during training.
    train_acc: dict, required, the training/validation accuracy during training.
    output_dir: str, optional, the output directory of the plot.
    close: bool, optional, whether to close the plot or not.

    Returns
    -------
    None
    """
    LOGGER.debug(f"Plotting the training/validation loss during training: {train_loss}")
    loss = pd.DataFrame(train_loss)
    loss.to_csv(f"{output_dir}/loss.csv", index=False)

    labels = Labels(
        title="Training/Validation Loss against Number of Epochs",
        xlabel="Number of Epochs",
        ylabel="Training/Validation Loss",
    )
    _, ax = initialize_plot(figsize=(10, 10), labels=labels)
    sns.lineplot(data=loss, ax=ax)
    savefig_and_close("lossHistory.jpg", output_dir, close)

    LOGGER.debug(
        f"Plotting the training/validation accuracy during training: {train_acc}"
    )
    acc = pd.DataFrame(train_acc)
    acc.to_csv(f"{output_dir}/acc.csv", index=False)

    labels = Labels(
        title="Training/Validation Accuracy against Number of Epochs",
        xlabel="Number of Epochs",
        ylabel="Training/Validation Accuracy",
    )
    _, ax = initialize_plot(figsize=(10, 10), labels=labels)
    sns.lineplot(data=acc, ax=ax)
    savefig_and_close("accHistory.jpg", output_dir, close)


def get_dataset_preview(
    dataset,
    mean: list[float],
    std: list[float],
    filename_remark="",
    output_dir=None,
    close=True,
) -> None:
    """
    Get the preview of the dataset.

    Parameters
    ----------
    dataset: torch.utils.data.Dataset, required, the dataset to be previewed.
    mean: list, required, the mean of the dataset.
    std: list, required, the standard deviation of the dataset.
    filename_remark: str, optional, the filename remark of the plot.
    output_dir: str, optional, the output directory of the plot.
    close: bool, optional, whether to close the plot or not.

    Returns
    -------
    None
    """
    nrows, ncols = 4, 4
    labels = Labels("Preview of Dataset")
    _, axes = initialize_plot(nrows, ncols, (10, 10), labels)
    images = iter(dataset)
    denormalizer = utils.preprocessing.Denormalize(np.array(mean), np.array(std))
    for row in range(nrows):
        for col in range(ncols):
            img = next(images)[0]
            img = denormalizer(img)
            img = img.numpy().transpose(1, 2, 0).astype(int)
            axes[row][col].imshow(img)
            axes[row][col].axis("off")
    savefig_and_close(f"datasetPreview_{filename_remark}.jpg", output_dir, close)
