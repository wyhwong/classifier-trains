import torch
from time import time
from copy import deepcopy

from .common import getLogger


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOGGER = getLogger("Training")
AVAILABLE_OPTIMIZER = ["sgd", "rmsprop", "adam", "adamw"]
AVAILABLE_SCHEDULER = ["step", "cosine"]


def trainModel(model, dataloaders, criterion, optimizer, scheduler, numEpochs, standard):
    trainingStart = time()
    LOGGER.info(f"Start time of training {trainingStart}")
    bestModel = model
    trainLoss = {"Train": [], "Val": []}
    trainAcc = {"Train": [], "Val": []}
    bestRecord = 0

    for epoch in range(numEpochs):
        LOGGER.info(f'Epoch {epoch}/{numEpochs - 1}')
        LOGGER.info('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                LOGGER.debug(f'The {epoch}-th epoch training started')
                model.train()
            else:
                LOGGER.debug(f'The {epoch}-th epoch validation started')
                model.eval()

            epochLoss = 0.0
            epochCorrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler:
                            scheduler.step()

                epochLoss += loss.item() * inputs.size(0)
                epochCorrects += torch.sum(preds == labels.data)

            epochLoss = epochLoss / len(dataloaders[phase].dataset)
            epochAcc = epochCorrects.double() / len(dataloaders[phase].dataset)

            LOGGER.info(f'{phase} Loss: {epochLoss:.4f} Acc: {epochAcc:.4f}')

            if phase == 'val':
                if standard == "loss" and epochLoss < bestRecord:
                    bestRecord = epochLoss
                    bestModel = deepcopy(model)
                if standard == "acc" and epochAcc > bestRecord:
                    bestRecord = epochAcc
                    bestModel = deepcopy(model)

            trainAcc[phase].append(epochAcc)
            trainLoss[phase].append(epochLoss)

    timeElapsed = time() - trainingStart
    LOGGER.info(f'Training complete in {timeElapsed // 60:.0f}m {timeElapsed % 60:.0f}s')
    LOGGER.info(f'Best val Acc: {bestRecord:4f}')

    return bestModel, model, trainLoss, trainAcc


def getOptimizer(optimizer="adam", lr=1e-3, momentum=0.9, weight_decay=0, alpha=0.99, betas=(0.9, 0.999)):
    if optimizer.lower() not in AVAILABLE_OPTIMIZER:
        raise NotImplementedError("The optimizer is not implemented in this TAO-like pytorch classifier.")
    if optimizer.lower() == "sgd":
        LOGGER.debug(f"Creating optimizer {optimizer}, {lr=}, {momentum=}, {weight_decay=}.")
        return torch.optim.SGD(lr=lr,
                               momentum=momentum,
                               weight_decay=weight_decay)
    elif optimizer.lower() == "rmsprop":
        LOGGER.debug(f"Creating optimizer {optimizer}, {lr=}, {momentum=}, {weight_decay=}, {alpha=}.")
        return torch.optim.RMSprop(lr=lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay,
                                   alpha=alpha)
    elif optimizer.lower() == "adam":
        LOGGER.debug(f"Creating optimizer {optimizer}, {betas=}, {weight_decay=}.")
        return torch.optim.Adam(lr=lr,
                                betas=betas,
                                weight_decay=weight_decay)
    elif optimizer.lower() == "adamw ":
        LOGGER.debug(f"Creating optimizer {optimizer}, {betas=}, {weight_decay=}.")
        return torch.optim.AdamW(lr=lr,
                                 betas=betas,
                                 weight_decay=weight_decay)


def getScheduler(scheduler, optimizer:torch.optim, numEpochs, step_size=30, gamma=0.1, minlr=0):
    if scheduler.lower() not in AVAILABLE_SCHEDULER:
        raise NotImplementedError("The scheduler is not implemented in this TAO-like pytorch classifier.")
    if scheduler.lower() == "step":
        LOGGER.debug(f"Creating scheduler {scheduler}, {step_size=}, {gamma=}.")
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                               step_size=step_size,
                                               gamma=gamma)
    elif scheduler.lower() == "cosine":
        LOGGER.debug(f"Creating scheduler {scheduler}, {numEpochs=}, {minlr=}")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                          T_max=numEpochs,
                                                          eta_min=minlr)


def getPreview(dataset):
    pass
