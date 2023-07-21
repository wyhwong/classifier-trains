import torch
import onnx

from .logger import get_logger

LOGGER = get_logger("utils/export")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    LOGGER.info(f"Exported the model: {export_path=}.")


def check_model_is_valid(model_path: str) -> None:
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    LOGGER.info(f"Checked onnx model at {model_path}.")
