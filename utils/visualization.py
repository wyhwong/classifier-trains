import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .common import getLogger

LOGGER = getLogger("Visualization")


def initializePlot(
        nrows=1, ncols=1, height=6, width=10, title="", xlabel="", ylabel="",
        tpad=2.5, lpad=0.1, bpad=0.12, fontsize=12
    ):
    LOGGER.debug(f"Initialize plt.subplots, {nrows=}, {ncols=}, {height=}, {width=}, {title=}, {xlabel=}, {ylabel=}, {tpad=}, {lpad=}, {bpad=}, {fontsize=}")
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    fig.tight_layout(pad=tpad)
    fig.subplots_adjust(left=lpad, bottom=bpad)
    fig.suptitle(title, fontsize=fontsize)
    fig.text(x=0.04, y=0.5, s=ylabel, fontsize=fontsize,
             rotation="vertical",verticalalignment='center')
    fig.text(x=0.5, y=0.04, s=xlabel, fontsize=fontsize,
             horizontalalignment='center')
    LOGGER.debug("Initialized subplots")
    return fig, axes


def visualizeAccAndLoss(trainLoss:dict, trainAcc:dict, outputDir:str, close=True) -> None:
    LOGGER.debug(f"Plotting the training/validation loss during training: {trainLoss}")
    loss = pd.DataFrame(trainLoss)
    _, ax = initializePlot(height=10,
                           width=10,
                           title="Training/Validation Loss against Number of Epochs",
                           xlabel="Number of Epochs",
                           ylabel="Training/Validation Loss")
    sns.lineplot(data=loss, ax=ax)
    plt.savefig(f"{outputDir}/lossHistory.jpg", facecolor="w")
    LOGGER.debug("Plotted the training/validation loss during training.")
    if close:
        LOGGER.debug("Closed the plot.")
        plt.close()

    LOGGER.debug(f"Plotting the training/validation accuracy during training: {trainAcc}")
    acc = pd.DataFrame(trainAcc)
    _, ax = initializePlot(height=10,
                           width=10,
                           title="Training/Validation Accuracy against Number of Epochs",
                           xlabel="Number of Epochs",
                           ylabel="Training/Validation Accuracy")
    sns.lineplot(data=acc, ax=ax)
    plt.savefig(f"{outputDir}/accHistory.jpg", facecolor="w")
    LOGGER.debug("Plotted the training/validation accuracy during training.")
    if close:
        LOGGER.debug("Closed the plot.")
        plt.close()


def getDatasetPreview(dataset, outputDir:str, filenameRemark="", close=True) -> None:
    plt.savefig(f"{outputDir}/datasetPreview_{filenameRemark}.jpg", facecolor="w")
    LOGGER.debug("Plotted the training/validation accuracy during training.")
    if close:
        LOGGER.debug("Closed the plot.")
        plt.close()
