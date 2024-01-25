import onnx
import torch
import torchvision

import env
import logger
import schemas.constants


local_logger = logger.get_logger(__name__)


def initialize_model(
    backbone: schemas.constants.ModelBackbone,
    weights: str,
    num_classes: int,
    unfreeze_all_params: bool,
) -> torchvision.models:
    """
    Initializes a model with the specified backbone, weights, and number of classes.

    Args:
    -----
        backbone (schemas.constants.ModelBackbone):
            The backbone architecture of the model.

        weights (str):
            The path to the pre-trained weights file.

        num_classes (int):
            The number of classes in the dataset.

        unfreeze_all_params (bool):
            Whether to unfreeze all parameters of the model.

    Returns:
    -----
        model (torchvision.models):
            The initialized model.
    """

    local_logger.info("Initializing model backbone: %s.", backbone.value)
    model = getattr(torchvision.models, backbone.value)(weights=weights)

    # Modify output layer to fit number of classes
    if "resnet" in backbone.value:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif "alexnet" in backbone.value:
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif "vgg" in backbone.value:
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif "squeezenet" in backbone.value:
        model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    elif "densenet" in backbone.value:
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif "inception" in backbone.value:
        model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise NotImplementedError("The model is not implemented in this TAO-like pytorch classifier.")

    local_logger.info("Modified the output layer of the model.")

    if unfreeze_all_params:
        unfreeze_all_params(model)

    # To enable PyTorch 2 compiler for optimized performance
    # (Only support CUDA capability >= 7.0)
    if env.DEVICE == "cuda" and torch.cuda.get_device_properties("cuda").major >= 7:
        local_logger.info("Enabled PyTorch 2 compiler for optimized performance.")
        model = torch.compile(model)

    return model


def unfreeze_all_params(model: torchvision.models) -> None:
    """
    Unfreezes all parameters in the model by setting `requires_grad` to True for each parameter.

    Args:
    -----
        model (torchvision.models):
            The model whose parameters need to be unfrozen.

    Returns:
    -----
        None
    """

    for param in model.parameters():
        param.requires_grad = True

    local_logger.info("Unfreezed all parameters in the model.")


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
    local_logger.info("Loaded weights from local file: %s", model)


def save_weights(weights: torch.nn.Module.state_dict, export_path: str) -> None:
    """
    Save the model weights to a file.

    Args:
    -----
        weights (torch.nn.Module.state_dict):
            The model weights to be saved.

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
        x.to(env.DEVICE),
        export_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    local_logger.info("Exported the model: %s.", export_path)


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
