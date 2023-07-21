import seaborn as sns
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from .visualization import Labels, initialize_plot, savefig_and_close
from .logger import get_logger

LOGGER = get_logger("utils/evaluation")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_model(model, dataloader, classes: list, output_dir=None, close=True) -> None:
    y_pred = []
    y_true = []

    model.to(DEVICE)
    model.eval()

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(inputs)

        outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs)

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

    metric = confusion_matrix(y_true, y_pred)
    metric_df = pd.DataFrame(
        metric / np.sum(metric, axis=1)[:, None],
        index=[i for i in classes],
        columns=[i for i in classes],
    )

    labels = Labels("Confusion Matrix", "Predicted Class", "Actual Class")
    _, ax = initialize_plot(figsize=(10, 10), labels=labels)
    sns.heatmap(metric_df, cmap="crest", annot=True, fmt=".1f", ax=ax)
    savefig_and_close("confusion_matrix.png", output_dir, close)
