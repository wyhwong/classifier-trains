import torch
import onnx

from .common import getLogger

LOGGER = getLogger("Export")


def exportModelToONNX(model, height, width, exportPath) -> None:
    LOGGER.debug(f"Changing model to evaluation mode.")
    model.eval()
    LOGGER.debug(f"Generating a random input for exporting.")
    x = torch.randn(1, 3, height, width, requires_grad=True)
    LOGGER.debug(f"Exporting the model: {exportPath=}.")
    torch.onnx.export(model,
                      x,
                      exportPath,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names = ['input'],
                      output_names = ['output'])
    LOGGER.info(f"Successfully exported the model: {exportPath=}.")


def checkModelIsValid(modelPath:str):
    LOGGER.info(f"Checking onnx model at {modelPath}")
    onnx_model = onnx.load(modelPath)
    onnx.checker.check_model(onnx_model)
    LOGGER.info(f"Checked onnx model at {modelPath}")
