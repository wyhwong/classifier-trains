import numpy as np

import core.preprocessing
import core.visualization.base as base
import schemas.visualization as schemas


def get_dataset_preview(
    dataset,
    mean: list[float],
    std: list[float],
    filename="preview.png",
    output_dir=None,
    close=True,
) -> None:
    nrows, ncols = 4, 4
    labels = schemas.Labels("Preview of Dataset")
    _, axes = base.initialize_plot(nrows, ncols, (10, 10), labels)
    images = iter(dataset)
    denormalizer = core.preprocessing.Denormalize(np.array(mean), np.array(std))
    for row in range(nrows):
        for col in range(ncols):
            img = next(images)[0]
            img = denormalizer(img)
            img = img.numpy().transpose(1, 2, 0).astype(int)
            axes[row][col].imshow(img)
            axes[row][col].axis("off")
    base.savefig_and_close(filename, output_dir, close)
