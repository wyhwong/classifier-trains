from typing import Any

import onnx
import torch
import torchvision

import pipeline.env
import pipeline.logger


local_logger = pipeline.logger.get_logger(__name__)


def load_model(model: torchvision.models, model_path: str) -> None:
    """
    Loads the weights of a PyTorch model from a given file path.

    Args:
    -----
        model (torchvision.models):
            The PyTorch model to load the weights into.

        model_path (str):
            The file path to the saved model weights.

    Returns:
    -----
        None
    """

    weights = torch.load(model_path)
    model.load_state_dict(weights)
    local_logger.info("Loaded weights from local file: %s.", model)


def save_weights(weights: dict[str, Any], export_path: str) -> None:
    """
    Save the model weights to a file.

    Args:
    -----
        weights (dict[str, Any]):
            The model weights to be saved. (state_dict)

        export_path (str):
            The path where the weights will be saved.

    Returns:
    -----
        None
    """

    torch.save(weights, export_path)
    local_logger.info("Saved model weights at %s.", export_path)


def export_model_to_onnx(
    model: torchvision.models,
    input_height: int,
    input_width: int,
    export_path: str,
) -> None:
    """
    Export the PyTorch model to ONNX format.

    Args:
    -----
        model (torchvision.models):
            The PyTorch model to be exported.

        input_height (int):
            The height of the input tensor.

        input_width (int):
            The width of the input tensor.

        export_path (str):
            The path to save the exported ONNX model.

    Returns:
    -----
        None
    """

    model.eval()
    x = torch.randn(1, 3, input_height, input_width, requires_grad=True)
    torch.onnx.export(
        model,
        x.to(pipeline.env.DEVICE),
        export_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    local_logger.info("Exported the model at %s.", export_path)


def check_model_is_valid(model_path: str) -> None:
    """
    Check if the ONNX model at the given path is valid.

    Args:
    -----
        model_path (str):
            The path to the ONNX model file.

    Returns:
    -----
        None
    """

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    local_logger.info("Checked ONNX model at %s.", model_path)
