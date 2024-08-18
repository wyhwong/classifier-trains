import enum


class OptimizerType(enum.StrEnum):
    """The optimizer type."""

    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAM = "adam"
    ADAMW = "adamw"


class SchedulerType(enum.StrEnum):
    """The scheduler type."""

    STEP = "step"
    COSINE = "cosine"


class Criterion(enum.StrEnum):
    """The best criteria."""

    LOSS = "loss"
    ACCURACY = "accuracy"


class InterpolationType(enum.StrEnum):
    """The interpolation type."""

    INTER_NEAREST = "inter_nearest"
    INTER_LINEAR = "inter_linear"
    INTER_AREA = "inter_area"
    INTER_CUBIC = "inter_cubic"
    INTER_LANCZOS4 = "inter_lanczos4"


class PaddingType(enum.StrEnum):
    """The padding type."""

    TOPLEFT = "top_left"
    TOPRIGHT = "top_right"
    BOTTOMLEFT = "bottom_left"
    BOTTOMRIGHT = "bottom_right"
    CENTER = "center"


class ModelBackbone(enum.StrEnum):
    """The model backbone."""

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


class Phase(enum.StrEnum):
    """The phase."""

    TRAINING = "train"
    VALIDATION = "val"
    TEST = "test"

    def __call__(self, suffix: str) -> str:
        """Return the phase with the suffix."""

        return f"{self.value}_{suffix}"
