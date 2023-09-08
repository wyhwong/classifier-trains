import torch
import torchvision

from tqdm import tqdm

import env


def predict(
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
