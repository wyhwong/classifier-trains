import enum


class OptimizerType(str, enum.Enum):
    """The optimizer type."""

    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAM = "adam"
    ADAMW = "adamw"


class SchedulerType(str, enum.Enum):
    """The scheduler type."""

    STEP = "step"
    COSINE = "cosine"


class Criterion(str, enum.Enum):
    """The best criteria."""

    LOSS = "loss"
    ACCURACY = "accuracy"


class InterpolationType(str, enum.Enum):
    """The interpolation type.
    These are based on the support of torchvision
    Details please check:
    https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.resize.html
    """

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest_exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


class PaddingType(str, enum.Enum):
    """The padding type."""

    TOPLEFT = "top_left"
    TOPRIGHT = "top_right"
    BOTTOMLEFT = "bottom_left"
    BOTTOMRIGHT = "bottom_right"
    CENTER = "center"


class ModelBackbone(str, enum.Enum):
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

    @classmethod
    def resnets(cls):
        """Return the resnet backbones."""

        return [cls.RESNET18, cls.RESNET34, cls.RESNET50, cls.RESNET152]

    @property
    def is_resnet(self) -> bool:
        """Check if the backbone is resnet."""

        return self in self.resnets()

    @classmethod
    def alexnets(cls):
        """Return the alexnet backbones."""

        return [cls.ALEXNET]

    @property
    def is_alexnet(self) -> bool:
        """Check if the backbone is alexnet."""

        return self in self.alexnets()

    @classmethod
    def vggs(cls):
        """Return the vgg backbones."""

        return [cls.VGG11, cls.VGG11_BN, cls.VGG13, cls.VGG13_BN, cls.VGG16, cls.VGG16_BN, cls.VGG19, cls.VGG19_BN]

    @property
    def is_vgg(self) -> bool:
        """Check if the backbone is vgg."""

        return self in self.vggs()

    @classmethod
    def squeezenets(cls):
        """Return the squeezenet backbones."""

        return [cls.SQUEEZENET1_0, cls.SQUEEZENET1_1]

    @property
    def is_squeezenet(self) -> bool:
        """Check if the backbone is squeezenet."""

        return self in self.squeezenets()

    @classmethod
    def densenets(cls):
        """Return the densenet backbones."""

        return [cls.DENSENET121, cls.DENSENET161, cls.DENSENET169, cls.DENSENET201]

    @property
    def is_densenet(self) -> bool:
        """Check if the backbone is densenet."""

        return self in self.densenets()

    @classmethod
    def inceptionnets(cls):
        """Return the inception backbones."""

        return [cls.INCEPTION_V3]

    @property
    def is_inception(self) -> bool:
        """Check if the backbone is inception."""

        return self in self.inceptionnets()


class Phase(str, enum.Enum):
    """The phase."""

    TRAINING = "train"
    VALIDATION = "val"
    TESTING = "test"

    def __call__(self, suffix: str) -> str:
        """Return the phase with the suffix."""

        return f"{self.value}_{suffix}"
