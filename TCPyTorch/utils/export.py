import torch
import torchvision
import onnx

import utils.logger
import utils.env

LOGGER = utils.logger.get_logger("utils/export")


def export_model_to_onnx(
    model: torchvision.models, input_height: int, input_width: int, export_path: str
) -> None:
    """
    Export the model to onnx format.

    Parameters
    ----------
    model: torchvision.models, required, the model to be exported.
    input_height: int, required, the input height of the model.
    input_width: int, required, the input width of the model.
    export_path: str, required, the export path of the model.

    Returns
    -------
    None
    """
    model.eval()
    x = torch.randn(1, 3, input_height, input_width, requires_grad=True)
    torch.onnx.export(
        model,
        x.to(utils.env.DEVICE),
        export_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    LOGGER.info("Exported the model: %s.", export_path)


def check_model_is_valid(model_path: str) -> None:
    """
    Check if the model is valid.

    Parameters
    ----------
    model_path: str, required, the model path.

    Returns
    -------
    None
    """
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    LOGGER.info("Checked onnx model at %s.", model_path)
