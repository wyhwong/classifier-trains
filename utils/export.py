import os
import torch
import onnx

from .logger import get_logger

LOGGER = get_logger("utils/export")
DEVICE = os.getenv("DEVICE")


def export_model_to_onnx(model, input_height: int, input_width: int, export_path: str) -> None:
    model.eval()
    x = torch.randn(1, 3, input_height, input_width, requires_grad=True)
    torch.onnx.export(
        model,
        x.to(DEVICE),
        export_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    LOGGER.info("Exported the model: %s.", export_path)


def check_model_is_valid(model_path: str) -> None:
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    LOGGER.info("Checked onnx model at %s.", model_path)
