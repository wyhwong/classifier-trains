from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import torch
import torchvision
from torch import nn

import pipeline.core.visualization.base as base
import pipeline.logger
import pipeline.schemas.visualization as schemas


local_logger = pipeline.logger.get_logger(__name__)


def loss_curve(
    df: pd.DataFrame,
    filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    close: bool = False,
):
    """Plots the training/validation loss curve against the number of epochs.

    Args:
        df (pd.DataFrame): The dataframe containing the loss data.
        filename (str, optional): The name of the output image file. Defaults to None.
        output_dir (str, optional): The directory to save the output files. Defaults to None.
        close (bool, optional): Whether to close the plot after saving. Defaults to False.

    Returns:
        tuple: The figure and axes objects of the loss curve plot
    """

    labels = schemas.Labels(
        title="Training/Validation Loss against Number of Epochs",
        xlabel="Number of Epochs",
        ylabel="Training/Validation Loss",
    )
    fig, ax = base.initialize_plot(figsize=(10, 10), labels=labels)
    sns.lineplot(data=df, ax=ax)
    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)


def accuracy_curve(
    df: pd.DataFrame,
    filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    close: bool = False,
):
    """Plots the training/validation accuracy curve against the number of epochs.

    Args:
        df (pd.DataFrame): The dataframe containing the accuracy data.
        filename (str, optional): The name of the output image file. Defaults to None.
        output_dir (str, optional): The directory to save the output files. Defaults to None.
        close (bool, optional): Whether to close the plot after saving. Defaults to False.

    Returns:
        tuple: The figure and axes objects of the accuracy curve
    """

    labels = schemas.Labels(
        title="Training/Validation Accuracy against Number of Epochs",
        xlabel="Number of Epochs",
        ylabel="Training/Validation Accuracy",
    )
    fig, ax = base.initialize_plot(figsize=(10, 10), labels=labels)
    sns.lineplot(data=df, ax=ax)
    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    close: bool = False,
):
    """Plots the confusion matrix.

    Args:
        y_true (list[str]): The true labels.
        y_pred (list[str]): The predicted labels.
        filename (str, optional): The name of the output image file. Defaults to None.
        output_dir (str, optional): The directory to save the output files. Defaults to None.
        close (bool, optional): Whether to close the plot after saving. Defaults to False.
    """

    classes = np.unique(y_true)
    metric = sklearn.metrics.confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(
        metric / np.sum(metric, axis=1)[:, None],
        index=[i for i in classes],
        columns=[i for i in classes],
    )

    labels = schemas.Labels(
        title="Confusion Matrix",
        xlabel="Predicted Class",
        ylabel="Actual Class",
    )
    fig, ax = base.initialize_plot(figsize=(10, 10), labels=labels)
    sns.heatmap(df, cmap="crest", annot=True, fmt=".1f", ax=ax)
    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)


def roc_curves(
    models: list[nn.Module],
    model_names: list[str],
    dataloader: torchvision.datasets.ImageFolder,
    mapping: dict[str, str],
    output_dir=None,
    close=True,
) -> None:
    """Plot the ROC curve for each class.

    Args:
        models (list[nn.Module]): The list of models.
        model_names (list[str]): The list of model names.
        dataloader (torch.utils.data.DataLoader): The data loader.
        mapping (dict[str, str]): The mapping of class indices to class names.
        output_dir (str, optional): The output directory. Defaults to None.
        close (bool, optional): Whether to close the plot after saving. Defaults to True.

    Raises:
        FileNotFoundError: If the output directory does not exist.
    """

    confidence: dict[str, np.ndarray] = {}
    y_true: dict[str, np.ndarray] = {}

    # Get the confidence and true labels for each model
    for idx, model in enumerate(models):
        model_name = model_names[idx]
        model_y_true, _, model_confidence = pipeline.core.model.inference.predict(model, dataloader)
        confidence[model_name] = np.array(model_confidence)
        # Change y_true to onehot format
        y_true_tensor = torch.tensor(model_y_true)
        y_true_tensor = y_true_tensor.reshape((y_true_tensor.shape[0], 1))
        _y_true_tensor = torch.zeros(y_true_tensor.shape[0], len(mapping))
        _y_true_tensor.scatter_(dim=1, index=y_true_tensor, value=1)
        y_true[model_name] = np.array(_y_true_tensor)

    # Plot the ROC curve for each class
    for obj_index, obj_class in enumerate(mapping):
        labels = schemas.Labels(f"ROC Curve ({obj_class})", "FPR", "TPR")
        _, ax = base.initialize_plot(figsize=(10, 10), labels=labels)
        for model_name in model_names:
            fpr, tpr, _ = sklearn.metrics.roc_curve(
                y_true[model_name][:, obj_index], confidence[model_name][:, obj_index]
            )
            auc = sklearn.metrics.roc_auc_score(y_true[model_name][:, obj_index], confidence[model_name][:, obj_index])
            sns.lineplot(
                x=fpr,
                y=tpr,
                label=f"{model_name}: AUC={auc:.4f}",
                ax=ax,
                errorbar=None,
            )

        sns.lineplot(x=[0, 1], y=[0, 1], ax=ax, linestyle="--", errorbar=None)
        ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05])
        plt.legend(loc="lower right")
        base.savefig_and_close(f"roc_curve_{obj_class}.png", output_dir, close)
