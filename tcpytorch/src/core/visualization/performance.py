from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
import sklearn.metrics

import core.model.inference
import core.visualization.base as base
import logger
import schemas.visualization as schemas


local_logger = logger.get_logger(__name__)


def loss_curve(
    df: pd.DataFrame,
    save_csv=True,
    filename="loss_history.jpg",
    output_dir: Optional[str] = None,
    close=True,
):
    """
    Plots the training/validation loss curve against the number of epochs.

    Args:
    -----
        df (pd.DataFrame):
            The DataFrame containing the loss values.

        save_csv (bool, optional):
            Whether to save the DataFrame as a CSV file. Defaults to True.

        filename (str, optional):
            The name of the output file. Defaults to "loss_history.jpg".

        output_dir (str, optional):
            The directory to save the output file. Defaults to None.

        close (bool, optional):
            Whether to close the plot after saving. Defaults to True.

    Returns:
    -----
        The figure and axes objects of the loss curve plot.
    """

    labels = schemas.Labels(
        title="Training/Validation Loss against Number of Epochs",
        xlabel="Number of Epochs",
        ylabel="Training/Validation Loss",
    )
    fig, ax = base.initialize_plot(figsize=(10, 10), labels=labels)
    sns.lineplot(data=df, ax=ax)
    if output_dir and save_csv:
        df.to_csv(f"{output_dir}/{filename.replace('jpg', 'csv')}", index=False)
    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)


def accuracy_curve(
    df: pd.DataFrame,
    save_csv=True,
    filename="accuracy_history.jpg",
    output_dir: Optional[str] = None,
    close=True,
):
    """
    Plots the training/validation accuracy curve against the number of epochs.

    Args:
    -----
        df (pd.DataFrame):
            The dataframe containing the accuracy data.

        save_csv (bool, optional):
            Whether to save the accuracy data as a CSV file. Defaults to True.

        filename (str, optional):
            The name of the output image file. Defaults to "accuracy_history.jpg".

        output_dir (str, optional):
            The directory to save the output files. Defaults to None.

        close (bool, optional):
            Whether to close the plot after saving. Defaults to True.

    Returns:
    -----
        The figure and axes objects of the loss curve plot.
    """

    labels = schemas.Labels(
        title="Training/Validation Accuracy against Number of Epochs",
        xlabel="Number of Epochs",
        ylabel="Training/Validation Accuracy",
    )
    fig, ax = base.initialize_plot(figsize=(10, 10), labels=labels)
    sns.lineplot(data=df, ax=ax)
    if output_dir and save_csv:
        df.to_csv(f"{output_dir}/{filename.replace('jpg', 'csv')}", index=False)
    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    filename="confusion_matrix.png",
    output_dir: Optional[str] = None,
    close=True,
):
    """
    Generate a confusion matrix plot based on the true and predicted labels.

    Args:
        y_true (list[str]):
            List of true labels.

        y_pred (list[str]):
            List of predicted labels.

        filename (str, optional):
            Name of the output file. Defaults to "confusion_matrix.png".

        output_dir (str, optional):
            Directory to save the output file. Defaults to None.

        close (bool, optional):
            Whether to close the plot after saving. Defaults to True.

    Returns:
    -----
        The figure and axes objects of the loss curve plot.
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
    models: list[torchvision.models],
    model_names: list[str],
    dataloader: torchvision.datasets.ImageFolder,
    classes: list,
    output_dir=None,
    close=True,
) -> None:
    """
    Generate ROC curve for multiple models.

    Args:
    -----
        models (list[torchvision.models]):
            List of models to generate ROC curves.

        model_names (list[str]):
            List of model names.

        dataloader (torchvision.datasets.ImageFolder):
            Dataloader for the dataset.

        classes (list):
            List of class labels.

        output_dir (str, optional):
            Output directory to save the ROC curve plots. Defaults to None.

        close (bool, optional):
            Flag to close the plot after saving. Defaults to True.

    Returns:
    -----
        None
    """

    confidence = {}
    y_true = {}

    # Get the confidence and true labels for each model
    for model in models:
        y_true[model_name], _, confidence[model_name] = core.model.inference.predict(model, dataloader)
        confidence[model_name] = np.array(confidence[model_name])
        # Change y_true to onehot format
        y_true_tensor = torch.tensor(y_true[model_name])
        y_true_tensor = y_true_tensor.reshape((y_true_tensor.shape[0], 1))
        y_true[model_name] = torch.zeros(y_true_tensor.shape[0], len(classes))
        y_true[model_name].scatter_(dim=1, index=y_true_tensor, value=1)
        y_true[model_name] = np.array(y_true[model_name])

    # Plot the ROC curve for each class
    for obj_index, obj_class in enumerate(classes):
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
