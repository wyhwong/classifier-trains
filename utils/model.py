import torch
import torchvision

from .logger import get_logger

LOGGER = get_logger("Model")
AVAILABLE_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet152",
    "alexnet",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "squeezenet1_0",
    "squeezenet1_1",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "inception_v3",
]


def initialize_model(arch: str, weights: str, num_classes: int, unfreeze_all_params: bool) -> torchvision.models:
    if arch.lower() in AVAILABLE_MODELS:
        LOGGER.info(f"Targeted arch {arch} in available models, extracting.")
        model = getattr(torchvision.models, arch)(weights=weights)
        LOGGER.debug(f"Extracted model: {model}.")

    LOGGER.debug("Modifying the output layer of the model.")
    if "resnet" in arch:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif "alexnet" in arch:
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif "vgg" in arch:
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif "squeezenet" in arch:
        model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
    elif "densenet" in arch:
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif "inception" in arch:
        model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise NotImplementedError("The model is not implemented in this TAO-like pytorch classifier.")
    LOGGER.debug("Modified the output layer of the model.")

    if unfreeze_all_params:
        model = unfreeze_all_params(model)
    return model


def unfreeze_all_params(model) -> None:
    for param in model.parameters():
        param.requires_grad = True
    LOGGER.info(f"Unfreezed all parameters in the model.")


def load_model(model, model_path: str) -> None:
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    LOGGER.info(f"Loaded weights from loacl file, {model}.")


def save_weights(weights, export_path: str) -> None:
    torch.save(weights, export_path)
    LOGGER.info(f"Saved model weights at {export_path}.")
