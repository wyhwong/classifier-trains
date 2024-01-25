import enum


class OptimizerType(enum.Enum):
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAM = "adam"
    ADAMW = "adamw"


class SchedulerType(enum.Enum):
    STEP = "step"
    COSINE = "cosine"


class BestCriteria(enum.Enum):
    LOSS = "loss"
    ACCURACY = "accuracy"


class InterpolationType(enum.Enum):
    INTER_NEAREST = "INTER_NEAREST"
    INTER_LINEAR = "INTER_LINEAR"
    INTER_AREA = "INTER_AREA"
    INTER_CUBIC = "INTER_CUBIC"
    INTER_LANCZOS4 = "INTER_LANCZOS4"


class PaddingType(enum.Enum):
    TOPLEFT = "top_left"
    TOPRIGHT = "top_right"
    BOTTOMLEFT = "bottom_left"
    BOTTOMRIGHT = "bottom_right"
    CENTER = "center"
    NONE = None


class ModelBackbone(enum.Enum):
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    RESNET152 = "resnet152"
    ALEXNET = "alexnet"
    VGG11 = "vgg11"
    VGG11_BN = "vgg11_bn"
    VGG13 = "vgg13"
    VGG13_BN = "vgg13_bn"
    VGG16 = "vgg16"
    VGG16_BN = "vgg16_bn"
    VGG19 = "vgg19"
    VGG19_BN = "vgg19_bn"
    SQUEEZENET1_0 = "squeezenet1_0"
    SQUEEZENET1_1 = "squeezenet1_1"
    DENSENET121 = "densenet121"
    DENSENET161 = "densenet161"
    DENSENET169 = "densenet169"
    DENSENET201 = "densenet201"
    INCEPTION_V3 = "inception_v3"
