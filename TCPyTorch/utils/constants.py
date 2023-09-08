# For training.py
AVAILABLE_OPTIMIZER = ["sgd", "rmsprop", "adam", "adamw"]
AVAILABLE_SCHEDULER = ["step", "cosine"]
AVAILABLE_STANDARD = ["loss", "acc"]

# For preprocessing
AVAILABLE_INTERPOLATION = [
    "INTER_NEAREST",
    "INTER_LINEAR",
    "INTER_AREA",
    "INTER_CUBIC",
    "INTER_LANCZOS4",
]
AVAILABLE_PADDING = ["topLeft", "topRight", "bottomLeft", "bottomRight", None]

# For model.py
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

# For logger.py
LOGFMT = "%(asctime)s [%(name)s | %(levelname)s]: %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
