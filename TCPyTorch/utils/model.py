import torch
import torchvision

import utils.logger
import utils.constants

LOGGER = utils.logger.get_logger("utils/model")


def initialize_model(
    arch: str, weights: str, num_classes: int, unfreeze_all_params: bool
) -> torchvision.models:
    """
    Initialize the model with the given arch, weights, num_classes, and unfreeze_all_params.

    Parameters
    ----------
    arch: str, required, the architecture of the model.
    weights: str, required, the weights of the model.
    num_classes: int, required, the number of classes of the model.
    unfreeze_all_params: bool, required, whether to unfreeze all parameters or not.

    Returns
    -------
    model: torchvision.models, the model.
    """
    if arch.lower() in utils.constants.AVAILABLE_MODELS:
        model = getattr(torchvision.models, arch)(weights=weights)
        LOGGER.info("Extracted model: %s", model)

    LOGGER.debug("Modifying the output layer of the model.")
    if "resnet" in arch:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif "alexnet" in arch:
        model.classifier[6] = torch.nn.Linear(
            model.classifier[6].in_features, num_classes
        )
    elif "vgg" in arch:
        model.classifier[6] = torch.nn.Linear(
            model.classifier[6].in_features, num_classes
        )
    elif "squeezenet" in arch:
        model.classifier[1] = torch.nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model.num_classes = num_classes
    elif "densenet" in arch:
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif "inception" in arch:
        model.AuxLogits.fc = torch.nn.Linear(
            model.AuxLogits.fc.in_features, num_classes
        )
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise NotImplementedError(
            "The model is not implemented in this TAO-like pytorch classifier."
        )
    LOGGER.info("Modified the output layer of the model.")

    if unfreeze_all_params:
        model = unfreeze_all_params(model)
    return model


def unfreeze_all_params(model: torchvision.models) -> None:
    """
    Unfreeze all parameters in the model.

    Parameters
    ----------
    model: torchvision.models, required, the model.

    Returns
    -------
    None
    """
    for param in model.parameters():
        param.requires_grad = True
    LOGGER.info("Unfreezed all parameters in the model.")


def load_model(model: torchvision.models, model_path: str) -> None:
    """
    Load the model from the given model_path.

    Parameters
    ----------
    model: torchvision.models, required, the model.
    model_path: str, required, the model path.

    Returns
    -------
    None
    """
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    LOGGER.info("Loaded weights from loacl file: %s", model)


def save_weights(weights: torch.nn.Module.state_dict, export_path: str) -> None:
    """
    Save the model weights at the given export_path.

    Parameters
    ----------
    weights: torch.nn.Module.state_dict, required, the weights of the model.
    export_path: str, required, the export path of the model weights.

    Returns
    -------
    None
    """
    torch.save(weights, export_path)
    LOGGER.info("Saved model weights at %s.", export_path)
