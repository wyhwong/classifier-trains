from typing import Callable, Optional

import pandas as pd
import torch

import pipeline.core.visualization.base as b
import pipeline.core.visualization.performance as p
from pipeline.schemas import constants, visualization


class Visualizer:
    """Class to visualize the data."""

    @staticmethod
    def plot_training_history(
        training_history: dict[constants.Criterion, dict[constants.Phase, list[float]]],
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

        return {
            constants.Criterion.LOSS: p.loss_curve(
                df=pd.DataFrame(training_history[constants.Criterion.LOSS]),
                filename=filename,
                output_dir=output_dir,
                close=close,
            ),
            constants.Criterion.ACCURACY: p.accuracy_curve(
                df=pd.DataFrame(training_history[constants.Criterion.ACCURACY]),
                filename=filename,
                output_dir=output_dir,
                close=close,
            ),
        }

    @staticmethod
    def plot_roc_curves(
        df: pd.DataFrame,
        filename: Optional[str] = None,
        output_dir: Optional[str] = None,
        close: bool = False,
    ):
        """Plots the ROC curves.

        Args:
            df (pd.DataFrame): The dataframe containing the ROC data.
            filename (str, optional): The name of the output image file. Defaults to None.
            output_dir (str, optional): The directory to save the output files. Defaults to None.
            close (bool, optional): Whether to close the plot after saving. Defaults to False.

        Returns:
            tuple: The figure and axes objects of the ROC curves
        """

    @staticmethod
    def plot_dataset_preview(
        dataset: torch.utils.data.Dataset,
        denorm_fn: Callable,
        filename: Optional[str] = None,
        output_dir: Optional[str] = None,
        close: bool = False,
    ):
        """Get preview of the dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to preview.
            mean (list[float]): The mean values.
            std (list[float]): The standard deviation values.
            filename (str, optional): The filename. Defaults to "preview.png".
            output_dir (str, optional): The output directory. Defaults to None.
            close (bool, optional): Whether to close the figure. Defaults to True.

        Returns:
            tuple: The figure and axes.
        """

        nrows, ncols = 4, 4
        labels = visualization.Labels("Preview of Dataset")
        fig, axes = b.initialize_plot(nrows, ncols, (10, 10), labels)
        images = iter(dataset)
        for row in range(nrows):
            for col in range(ncols):
                img = next(images)[0]
                img = denorm_fn(img)
                img = img.numpy().transpose(1, 2, 0).astype(int)
                axes[row][col].imshow(img)
                axes[row][col].axis("off")
        b.savefig_and_close(filename, output_dir, close)
        return (fig, axes)
