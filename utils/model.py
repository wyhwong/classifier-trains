import torchvision
import torch.nn as nn

from .common import getLogger

LOGGER = getLogger("Model")
AVAILABLE_MODELS = ["resnet18", "resnet34", "resnet50", "resnet152",
                    "alexnet",
                    "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
                    "squeezenet1_0", "squeezenet1_1",
                    "densenet121", "densenet161", "densenet169", "densenet201", "inception_v3"]


def initializeModel(arch, weights, numClasses):
    if arch in AVAILABLE_MODELS:
        model = getattr(torchvision.models, arch)(weights=weights)

    if "resnet" in arch:
        model.fc = nn.Linear(model.fc.in_features, numClasses)
    elif "alexnet" in arch:
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, numClasses)
    elif "vgg" in arch:
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, numClasses)
    elif "squeezenet" in arch:
        model.classifier[1] = nn.Conv2d(512, numClasses, kernel_size=(1,1), stride=(1,1))
        model.num_classes = numClasses
    elif "densenet" in arch:
        model.classifier = nn.Linear(model.classifier.in_features, numClasses)
    elif "inception" in arch:
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, numClasses)
        model.fc = nn.Linear(model.fc.in_features, numClasses)
    else:
        raise NotImplementedError("The model is not implemented in this TAO-like pytorch classifier.")
    return model


def unfreezeAllParams(model):
    LOGGER.info(f'Unfreezing all parameters in the model')
    for param in model.parameters():
        param.requires_grad = True
    LOGGER.info(f'Unfreezed all parameters in the model')
