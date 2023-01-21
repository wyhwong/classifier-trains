import cv2
import numpy as np
from torchvision import transforms

from .common import getLogger

LOGGER = getLogger("Preprocessing")
AVAILABLE_INTERPOLATION = ["INTER_NEAREST", "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"]
AVAILABLE_PADDING = ["topLeft", "topRight", "bottomLeft", "bottomRight", None]


def getTransforms(
        width:int, height:int, interpolation:str, maintainAspectRatio:bool,
        padding:str,mean:list, std:list, hflipProb:float, vflipProb:float,
        maxRotate:float, centorCrop:bool, randomCrop:bool, grayScale:bool,
        randomColorAugmentation: bool
    ) -> dict:

    spatialAugmentation = []
    colorAugmentation = []
    resizeAndPadding = [PilToCV2(),
                        Resize(width, height, interpolation, maintainAspectRatio, padding)]
    normalization = [transforms.ToTensor(),
                     transforms.Normalize(mean, std)]

    if hflipProb > 0:
        LOGGER.debug(f"Spatial augmentation added: {hflipProb=}.")
        spatialAugmentation.append(transforms.RandomHorizontalFlip(hflipProb))
    if vflipProb > 0:
        LOGGER.debug(f"Spatial augmentation added: {vflipProb=}.")
        spatialAugmentation.append(transforms.RandomVerticalFlip(vflipProb))
    if maxRotate > 0:
        LOGGER.debug(f"Spatial augmentation added: {maxRotate=}.")
        spatialAugmentation.append(transforms.RandomRotation(maxRotate))
    if centorCrop:
        LOGGER.debug(f"Spatial augmentation added: center crop.")
        spatialAugmentation.append(transforms.CenterCrop((height, width)))
    if randomCrop:
        LOGGER.debug(f"Spatial augmentation added: random crop.")
        spatialAugmentation.append(transforms.RandomCrop((height, width)))

    if grayScale:
        LOGGER.debug(f"Color augmentation added: grayscale.")
        colorAugmentation.append(transforms.Grayscale(3))
    if randomColorAugmentation:
        brightness, hue = 0.5, 0.3
        LOGGER.debug(f"Color augmentation added: coloer jitter with {brightness=}, {hue=}.")
        colorAugmentation.append(transforms.ColorJitter(brightness=brightness, hue=hue))

    LOGGER.info(f"Constructing transforms.compose for trainset: \n\t{spatialAugmentation=}, \n\t{colorAugmentation=}, \n\t{resizeAndPadding=}, \n\t{normalization=}.")
    data_transforms = {
        'train': transforms.Compose(
            spatialAugmentation + colorAugmentation + resizeAndPadding + normalization
        ),
        'val': transforms.Compose([
            resizeAndPadding + normalization
        ]),
    }
    LOGGER.info(f"Constructed transfroms.compose.")
    return data_transforms


class Resize:
    def __init__(self, width:int, height:int, interpolation:str, maintainAspectRatio:bool, padding:str) -> None:
        if width != height:
            raise NotImplementedError("Resizing with non-squared output size is not implemented.")
        if padding not in AVAILABLE_PADDING:
            raise NotImplementedError(f"Resizing with {padding} padding is invalid or not implemented.")
        if interpolation.upper() not in AVAILABLE_INTERPOLATION:
            raise NotImplementedError(f"Resizing with {interpolation} interpolation is invalid or not implemented.")
        self.width = width
        self.height = height
        self.dim = (width, height)
        self.interpolation = getattr(cv2, interpolation.upper())
        self.maintainAspectRatio = maintainAspectRatio
        self.padding = padding

    def __call__(self, image:np.ndarray) -> np.ndarray:
        if not self.maintainAspectRatio:
            return cv2.resize(image, (self.height, self.width), interpolation=self.interpolation)
        else:
            height, width, _ = image.shape
            if height > width:
                width = int(width / height * self.height)
                image = cv2.resize(image, (width, self.height), interpolation=self.interpolation)
            else:
                height = int(height / width * self.width)
                image = cv2.resize(image, (self.width, height), interpolation=self.interpolation)
            outputImage = np.zeros((self.height, self.width, 3), dtype=float)
            if self.padding == "bottomRight":
                outputImage[:height,:width,] = image
            elif self.padding == "bottomLeft":
                outputImage[:height,self.width-width:,] = image
            elif self.padding == "topLeft":
                outputImage[self.height-height:,:width,] = image
            else:
                outputImage[self.height-height:,self.width-width:,] = image
            return outputImage


class PilToCV2:
    def __call__(self, image):
        return np.array(image)
