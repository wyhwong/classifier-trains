import torch
import torchvision

from .common import getLogger

LOGGER = getLogger("Model")
AVAILABLE_MODELS = ["resnet18", "resnet34", "resnet50", "resnet152",
                    "alexnet",
                    "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
                    "squeezenet1_0", "squeezenet1_1",
                    "densenet121", "densenet161", "densenet169", "densenet201", "inception_v3"]


def initializeModel(arch:str, weights:str, numClasses:int, unfreezeAllParams:bool) -> torchvision.models:
    if arch.lower() in AVAILABLE_MODELS:
        LOGGER.info(f"Targeted arch {arch} in available models, extracting.")
        model = getattr(torchvision.models, arch)(weights=weights)
        LOGGER.debug(f"Extracted model: {model}.")

    LOGGER.debug("Modifying the output layer of the model.")
    if "resnet" in arch:
        model.fc = torch.nn.Linear(model.fc.in_features, numClasses)
    elif "alexnet" in arch:
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, numClasses)
    elif "vgg" in arch:
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, numClasses)
    elif "squeezenet" in arch:
        model.classifier[1] = torch.nn.Conv2d(512, numClasses, kernel_size=(1,1), stride=(1,1))
        model.num_classes = numClasses
    elif "densenet" in arch:
        model.classifier = torch.nn.Linear(model.classifier.in_features, numClasses)
    elif "inception" in arch:
        model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, numClasses)
        model.fc = torch.nn.Linear(model.fc.in_features, numClasses)
    else:
        raise NotImplementedError("The model is not implemented in this TAO-like pytorch classifier.")
    LOGGER.debug("Modified the output layer of the model.")

    if unfreezeAllParams:
        model = unfreezeAllParams(model)
    return model


def unfreezeAllParams(model) -> None:
    LOGGER.info(f'Unfreezing all parameters in the model.')
    for param in model.parameters():
        param.requires_grad = True
    LOGGER.info(f'Unfreezed all parameters in the model.')


def loadModel(model, modelPath:str) -> None:
    LOGGER.info(f"Loading weights from local file: {modelPath}.")
    weights = torch.load(modelPath)
    model.load_state_dict(weights)
    LOGGER.info(f"Loaded weights from loacl file, {model}.")


def saveWeights(weights, modelPath:str) -> None:
    LOGGER.info(f"Saving model weights at {modelPath}.")
    torch.save(weights, modelPath)
    LOGGER.info(f"Saved model weights.")
