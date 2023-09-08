import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

import utils.model
import utils.visualization
import utils.logger
import utils.env

LOGGER = utils.logger.get_logger("utils/evaluation")


def get_predictions(
    model: torchvision.models, dataloader: torchvision.datasets.ImageFolder
) -> tuple:
    """
    Get the predictions of the model with the given dataloader.

    Parameters
    ----------
    model: torchvision.models, required, the model to be evaluated.
    dataloader: torchvision.datasets.ImageFolder, required, the dataloader to be evaluated.

    Returns
    -------
    y_true: list, the true labels.
    y_pred: list, the predicted labels.
    y_pred_prob: list, the predicted probabilities.
    """
    y_pred = []
    y_pred_prob = []
    y_true = []

    model.to(utils.env.DEVICE)
    model.eval()

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(utils.env.DEVICE)
        labels = labels.to(utils.env.DEVICE)

        outputs = model(inputs.float())
        y_pred_prob.extend(outputs.detach().cpu().numpy())

        outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs)

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

    return (y_true, y_pred, y_pred_prob)


def evaluate_model(
    model: torchvision.models,
    dataloader: torchvision.datasets.ImageFolder,
    classes: list,
    output_dir=None,
    close=True,
) -> None:
    """
    Evaluate the model with the given dataloader.

    Parameters
    ----------
    model: torchvision.models, required, the model to be evaluated.
    dataloader: torchvision.datasets.ImageFolder, required, the dataloader to be evaluated.
    classes: list, required, the classes of the model.
    output_dir: str, optional, the output directory of the plot.
    close: bool, optional, whether to close the plot or not.

    Returns
    -------
    None
    """
    y_true, y_pred, _ = get_predictions(model, dataloader)
    metric = confusion_matrix(y_true, y_pred)
    metric_df = pd.DataFrame(
        metric / np.sum(metric, axis=1)[:, None],
        index=[i for i in classes],
        columns=[i for i in classes],
    )

    labels = utils.visualization.Labels(
        "Confusion Matrix", "Predicted Class", "Actual Class"
    )
    _, ax = utils.visualization.initialize_plot(figsize=(10, 10), labels=labels)
    sns.heatmap(metric_df, cmap="crest", annot=True, fmt=".1f", ax=ax)
    utils.visualization.savefig_and_close("confusion_matrix.png", output_dir, close)


def visualize_roc_curve(
    model_configs: list[dict[str, str]],
    dataloader: torchvision.datasets.ImageFolder,
    classes: list,
    output_dir=None,
    close=True,
) -> None:
    """
    Visualize the ROC curve of the model with the given dataloader.

    Parameters
    ----------
    model_configs: list, required, the model configurations.
    dataloader: torchvision.datasets.ImageFolder, required, the dataloader to be evaluated.
    classes: list, required, the classes of the model.
    output_dir: str, optional, the output directory of the plot.
    close: bool, optional, whether to close the plot or not.

    Returns
    -------
    None
    """
    y_pred_prob = {}
    y_true = {}
    for model_config in model_configs:
        model_name = model_config["name"]
        model = utils.model.initialize_model(
            model_config["arch"], "DEFAULT", len(classes), False
        )
        utils.model.load_model(model, model_config["path"])

        y_true[model_name], _, y_pred_prob[model_name] = get_predictions(
            model, dataloader
        )
        y_pred_prob[model_name] = np.array(y_pred_prob[model_name])
        # Change y_true to onehot format
        y_true_tensor = torch.tensor(y_true[model_name])
        y_true_tensor = y_true_tensor.reshape((y_true_tensor.shape[0], 1))
        y_true[model_name] = torch.zeros(y_true_tensor.shape[0], len(classes))
        y_true[model_name].scatter_(dim=1, index=y_true_tensor, value=1)
        y_true[model_name] = np.array(y_true[model_name])

    for obj_index, obj_class in enumerate(classes):
        labels = utils.visualization.Labels(f"ROC Curve ({obj_class})", "FPR", "TPR")
        _, ax = utils.visualization.initialize_plot(figsize=(10, 10), labels=labels)
        for model_config in model_configs:
            model_name = model_config["name"]
            fpr, tpr, _ = roc_curve(
                y_true[model_name][:, obj_index], y_pred_prob[model_name][:, obj_index]
            )
            auc = roc_auc_score(
                y_true[model_name][:, obj_index], y_pred_prob[model_name][:, obj_index]
            )
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
        utils.visualization.savefig_and_close(
            f"roc_curve_{obj_class}.png", output_dir, close
        )
