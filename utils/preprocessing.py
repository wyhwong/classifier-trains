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
        spatialAugmentation.append(transforms.RandomHorizontalFlip(hflipProb))
    if vflipProb > 0:
        spatialAugmentation.append(transforms.RandomVerticalFlip(vflipProb))
    if maxRotate > 9:
        spatialAugmentation.append(transforms.RandomRotation(maxRotate))
    if centorCrop:
        spatialAugmentation.append(transforms.CenterCrop((height, width)))
    if randomCrop:
        spatialAugmentation.append(transforms.RandomCrop((height, width)))

    if grayScale:
        colorAugmentation.append(transforms.Grayscale(3))
    if randomColorAugmentation:
        colorAugmentation.append(transforms.ColorJitter(brightness=.5, hue=.3))

    data_transforms = {
        'train': transforms.Compose(
            spatialAugmentation + colorAugmentation + resizeAndPadding + normalization
        ),
        'val': transforms.Compose([
            resizeAndPadding + normalization
        ]),
    }
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
                width = int(height / self.height * self.width)
            else:
                height = int(width / self.width * self.height)
            image = cv2.resize(image, (width, height), interpolation=self.interpolation)
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
    def __call__(image):
        return np.array(image)
