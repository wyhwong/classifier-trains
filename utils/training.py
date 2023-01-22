import torch
from datetime import datetime
from copy import deepcopy

from .common import getLogger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOGGER = getLogger("Training")
AVAILABLE_OPTIMIZER = ["sgd", "rmsprop", "adam", "adamw"]
AVAILABLE_SCHEDULER = ["step", "cosine"]


def trainModel(model, dataloaders, criterion, optimizer, scheduler, numEpochs, standard):
    trainingStart = datetime.now()
    LOGGER.info(f"Start time of training {trainingStart}.")
    LOGGER.info(f"Training using device: {DEVICE}")
    bestWeights = deepcopy(model.state_dict())
    trainLoss = {"train": [], "val": []}
    trainAcc = {"train": [], "val": []}
    bestRecord = 0

    for epoch in range(1, numEpochs+1):
        LOGGER.info(f'Epoch {epoch}/{numEpochs}')
        LOGGER.info('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                LOGGER.debug(f'The {epoch}-th epoch training started.')
                model.train()
            else:
                LOGGER.debug(f'The {epoch}-th epoch validation started.')
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
                            LOGGER.info(f"Learning rate in this epoch: {scheduler.get_last_lr()}.")

                epochLoss += loss.item() * inputs.size(0)
                epochCorrects += torch.sum(preds == labels.data)

            epochLoss = epochLoss / len(dataloaders[phase].dataset)
            epochAcc = epochCorrects.double() / len(dataloaders[phase].dataset)

            LOGGER.info(f"{phase} Loss: {epochLoss:.4f} Acc: {epochAcc:.4f}.")

            if phase == 'val':
                if standard == "loss" and epochLoss < bestRecord:
                    LOGGER.info(f"New Record: {epochLoss=} < {bestRecord}.")
                    bestRecord = epochLoss
                    bestWeights = deepcopy(model.state_dict())
                    LOGGER.debug(f"Updated best models.")
                if standard == "acc" and epochAcc > bestRecord:
                    LOGGER.info(f"New Record: {epochAcc=} > {bestRecord}.")
                    bestRecord = epochAcc
                    bestWeights = deepcopy(model.state_dict())
                    LOGGER.debug(f"Updated best models.")

            trainAcc[phase].append(epochAcc)
            trainLoss[phase].append(epochLoss)
            LOGGER.debug(f"Updated {trainAcc=}, {trainLoss=}.")

    lastWeights = deepcopy(model.state_dict())
    trainingEnd = datetime.now()
    timeElapsed = (trainingEnd - trainingStart).total_seconds()
    LOGGER.info(f"Training complete at {trainingEnd}")
    LOGGER.info(f'Training complete in {timeElapsed // 60:.0f}m {timeElapsed % 60:.0f}s.')
    LOGGER.info(f'Best val {standard}: {bestRecord:4f}.')
    return model, bestWeights, lastWeights, trainLoss, trainAcc


def getOptimizer(params, name="adam", lr=1e-3, momentum=0.9, weight_decay=0, alpha=0.99, betas=(0.9, 0.999)):
    if name.lower() not in AVAILABLE_OPTIMIZER:
        raise NotImplementedError("The optimizer is not implemented in this TAO-like pytorch classifier.")
    if name.lower() == "sgd":
        LOGGER.debug(f"Creating optimizer {name}, {lr=}, {momentum=}, {weight_decay=}.")
        return torch.optim.SGD(params,
                               lr=lr,
                               momentum=momentum,
                               weight_decay=weight_decay)
    elif name.lower() == "rmsprop":
        LOGGER.debug(f"Creating optimizer {name}, {lr=}, {momentum=}, {weight_decay=}, {alpha=}.")
        return torch.optim.RMSprop(params,
                                   lr=lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay,
                                   alpha=alpha)
    elif name.lower() == "adam":
        LOGGER.debug(f"Creating optimizer {name}, {betas=}, {weight_decay=}.")
        return torch.optim.Adam(params,
                                lr=lr,
                                betas=betas,
                                weight_decay=weight_decay)
    elif name.lower() == "adamw ":
        LOGGER.debug(f"Creating optimizer {name}, {betas=}, {weight_decay=}.")
        return torch.optim.AdamW(params,
                                 lr=lr,
                                 betas=betas,
                                 weight_decay=weight_decay)


def getScheduler(name:str, optimizer:torch.optim, numEpochs:int, stepSize=30, gamma=0.1, lrMin=0):
    if name.lower() not in AVAILABLE_SCHEDULER:
        raise NotImplementedError("The scheduler is not implemented in this TAO-like pytorch classifier.")
    if name.lower() == "step":
        LOGGER.debug(f"Creating scheduler {name}, {stepSize=}, {gamma=}.")
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                               step_size=stepSize,
                                               gamma=gamma)
    elif name.lower() == "cosine":
        LOGGER.debug(f"Creating scheduler {name}, {numEpochs=}, {lrMin=}")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                          T_max=numEpochs,
                                                          eta_min=lrMin)


def getPreview(dataset):
    pass
