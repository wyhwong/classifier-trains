from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm

import core.model
import core.visualization.base as base
import env
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
    metric = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(
        metric / np.sum(metric, axis=1)[:, None],
        index=[i for i in classes],
        columns=[i for i in classes],
    )
    labels = schemas.Labels("Confusion Matrix", "Predicted Class", "Actual Class")
    fig, ax = base.initialize_plot(figsize=(10, 10), labels=labels)
    sns.heatmap(df, cmap="crest", annot=True, fmt=".1f", ax=ax)
    base.savefig_and_close(filename, output_dir, close)
    return (fig, ax)


def roc_curves(
    model_configs: list[dict[str, str]],
    dataloader: torchvision.datasets.ImageFolder,
    classes: list,
    output_dir=None,
    close=True,
):
    """
    Generate ROC curve for multiple models.

    Args:
        model_configs (list[dict[str, str]]):
            List of model configurations.

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
        The figure and axes objects of the loss curve plot.
    """

    confidence = {}
    y_true = {}
    for model_config in model_configs:
        model_name = model_config["name"]
        model = core.model.initialize_model(model_config["arch"], "DEFAULT", len(classes), False)
        core.model.load_model(model, model_config["path"])

        y_true[model_name], _, confidence[model_name] = _get_predictions(model, dataloader)
        confidence[model_name] = np.array(confidence[model_name])
        # Change y_true to onehot format
        y_true_tensor = torch.tensor(y_true[model_name])
        y_true_tensor = y_true_tensor.reshape((y_true_tensor.shape[0], 1))
        y_true[model_name] = torch.zeros(y_true_tensor.shape[0], len(classes))
        y_true[model_name].scatter_(dim=1, index=y_true_tensor, value=1)
        y_true[model_name] = np.array(y_true[model_name])

    for obj_index, obj_class in enumerate(classes):
        labels = schemas.Labels(f"ROC Curve ({obj_class})", "FPR", "TPR")
        _, ax = base.initialize_plot(figsize=(10, 10), labels=labels)
        for model_config in model_configs:
            model_name = model_config["name"]
            fpr, tpr, _ = roc_curve(y_true[model_name][:, obj_index], confidence[model_name][:, obj_index])
            auc = roc_auc_score(y_true[model_name][:, obj_index], confidence[model_name][:, obj_index])
            sns.lineplot(
                x=fpr,
                y=tpr,
                label=f"{model_config['name']}: AUC={auc:.4f}",
                ax=ax,
                errorbar=None,
            )

        sns.lineplot(x=[0, 1], y=[0, 1], ax=ax, linestyle="--", errorbar=None)
        ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05])
        plt.legend(loc="lower right")
        base.savefig_and_close(f"roc_curve_{obj_class}.png", output_dir, close)


def _get_predictions(
    model: torchvision.models,
    dataloader: torchvision.datasets.ImageFolder,
) -> tuple[list, list, list]:
    """
    Get predictions from a trained model.

    Args:
    -----
        model (torchvision.models):
            The trained model.

        dataloader (torchvision.datasets.ImageFolder):
            The dataloader for the dataset.

    Returns:
    -----
        y_true (list):
            true labels

        y_pred (list):
            predicted labels

        confidence (list):
            predicted probabilities
    """

    y_pred = []
    confidence = []
    y_true = []

    model.to(env.DEVICE)
    model.eval()

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(env.DEVICE)
        labels = labels.to(env.DEVICE)

        outputs = model(inputs.float())
        confidence.extend(outputs.detach().cpu().numpy())

        outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs)

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

    return (y_true, y_pred, confidence)
