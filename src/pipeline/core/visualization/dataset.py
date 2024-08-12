import numpy as np
import torch

import pipeline.core.data.preprocessing
import pipeline.core.visualization.base as base
import pipeline.schemas.visualization as schemas


def get_dataset_preview(
    dataset: torch.utils.data.Dataset,
    mean: list[float],
    std: list[float],
    filename="preview.png",
    output_dir=None,
    close=True,
):
    """
    Get preview of the dataset.

    Args:
    -----
        dataset: torch.utils.data.Dataset
            Dataset to visualize.

        mean: list[float]
            Mean of the dataset.

        std: list[float]
            Standard deviation of the dataset.

        filename: str, optional
            Filename of the preview.

        output_dir: str, optional
            Output directory of the preview.

        close: bool, optional
            Close the figure after saving.

    Returns:
    --------
        fig: matplotlib.pyplot.figure
            Figure of the preview.

        axes: matplotlib.pyplot.axes
            Axes of the preview.
    """

    nrows, ncols = 4, 4
    labels = schemas.Labels("Preview of Dataset")
    fig, axes = base.initialize_plot(nrows, ncols, (10, 10), labels)
    images = iter(dataset)
    denormalizer = pipeline.core.data.preprocessing.Denormalize(np.array(mean), np.array(std))
    for row in range(nrows):
        for col in range(ncols):
            img = next(images)[0]
            img = denormalizer(img)
            img = img.numpy().transpose(1, 2, 0).astype(int)
            axes[row][col].imshow(img)
            axes[row][col].axis("off")
    base.savefig_and_close(filename, output_dir, close)

    return (fig, axes)
