import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from dataclasses import dataclass

from .common import check_and_create_dir
from .logger import get_logger
from .preprocessing import Denormalize

LOGGER = get_logger("Visualization")


@dataclass
class Padding:
    tpad: float = 2.5
    lpad: float = 0.1
    bpad: float = 0.12


@dataclass
class Labels:
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""


def initialize_plot(nrows=1, ncols=1, figsize=(10, 6), labels=Labels(), padding=Padding(), fontsize=12):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.tight_layout(pad=padding.tpad)
    fig.subplots_adjust(left=padding.lpad, bottom=padding.bpad)
    fig.suptitle(labels.title, fontsize=fontsize)
    fig.text(x=0.04, y=0.5, s=labels.ylabel, fontsize=fontsize, rotation="vertical", verticalalignment="center")
    fig.text(x=0.5, y=0.04, s=labels.xlabel, fontsize=fontsize, horizontalalignment="center")
    return fig, axes


def savefig_and_close(plot_filename: str, output_dir=None, savefig=False, close=True) -> None:
    if savefig:
        if output_dir is None:
            raise ValueError(f"output_dir must not be empty if savefig is True.")
        check_and_create_dir(dirpath=output_dir)
        savepath = f"{output_dir}/{plot_filename}"
        plt.savefig(savepath, facecolor="w")
        LOGGER.debug(f"Saved plot at {savepath}.")
    if close:
        plt.close()
        LOGGER.debug("Closed plot.")


def visualize_acc_and_loss(train_loss: dict, train_acc: dict, output_dir=None, savefig=False, close=True) -> None:
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
    savefig_and_close("lossHistory.jpg", output_dir, savefig, close)

    LOGGER.debug(f"Plotting the training/validation accuracy during training: {train_acc}")
    acc = pd.DataFrame(train_acc)
    acc.to_csv(f"{output_dir}/acc.csv", index=False)

    labels = Labels(
        title="Training/Validation Accuracy against Number of Epochs",
        xlabel="Number of Epochs",
        ylabel="Training/Validation Accuracy",
    )
    _, ax = initialize_plot(figsize=(10, 10), labels=labels)
    sns.lineplot(data=acc, ax=ax)
    savefig_and_close("accHistory.jpg", output_dir, savefig, close)


def get_dataset_preview(dataset, mean, std, filename_remark="", output_dir=None, savefig=False, close=True) -> None:
    nrows, ncols = 4, 4
    labels = Labels(title="Preview of Dataset")
    _, axes = initialize_plot(nrows=nrows, ncols=ncols, figsize=(10, 10), labels=labels)
    images = iter(dataset)
    denormalizer = Denormalize(mean, std)
    for row in range(nrows):
        for col in range(ncols):
            img = next(images)[0]
            img = denormalizer(img)
            img = img.numpy().transpose(1, 2, 0).astype(int)
            axes[row][col].imshow(img)
            axes[row][col].axis("off")
    savefig_and_close(f"datasetPreview_{filename_remark}.jpg", output_dir, savefig, close)
