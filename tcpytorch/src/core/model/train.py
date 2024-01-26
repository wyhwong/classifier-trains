from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torchvision
from tqdm import tqdm

import env
import logger
import schemas.constants


local_logger = logger.get_logger(__name__)


def train_model(
    model: torchvision.models,
    dataloaders: dict[str, torchvision.datasets.ImageFolder],
    criterion: torch.nn.Module,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    num_epochs: int,
    best_criteria: schemas.constants.BestCriteria,
) -> tuple:
    """
    Train the model.

    Args:
    -----
        model: torchvision.models
            Model to train.

        dataloaders: dict[str, torchvision.datasets.ImageFolder]
            Dataloaders for training and validation.

        criterion: torch.nn.Module
            Criterion for the model.

        optimizer: torch.optim
            Optimizer for the model.

        scheduler: torch.optim.lr_scheduler
            Scheduler for the optimizer.

        num_epochs: int
            Number of epochs to train.

        best_criteria: schemas.constants.BestCriteria
            Best criteria to save the best model.

    Returns:
    --------
        model: torchvision.models
            Trained model.

        best_weights: dict
            Best weights of the model.

        last_weights: dict
            Last weights of the model.

        train_loss: dict[str, list]
            Training loss of the model.

        train_acc: dict[str, list]
            Training accuracy of the model.
    """

    training_start = datetime.now()
    local_logger.info("Start time of training: %s", training_start)
    local_logger.info("Training using device: %s", env.DEVICE)

    best_weights = deepcopy(model.state_dict())
    train_loss: dict[str, list] = {"train": [], "val": []}
    train_acc: dict[str, list] = {"train": [], "val": []}
    best_record = np.inf if best_criteria is schemas.constants.BestCriteria.LOSS else -np.inf

    for epoch in range(1, num_epochs + 1):
        local_logger.info("-" * 40)
        local_logger.info("Epoch %d/%d", epoch, num_epochs)
        local_logger.info("-" * 20)

        for phase in [schemas.constants.Phase.TRAINING, schemas.constants.Phase.VALIDATION]:
            if phase is schemas.constants.Phase.TRAINING:
                local_logger.debug("The %d-th epoch training started.", epoch)
                model.train()
            else:
                local_logger.debug("The %d-th epoch validation started.", epoch)
                model.eval()

            epoch_loss = np.inf
            epoch_corrects = 0
            for inputs, labels in tqdm(dataloaders[phase.value]):
                inputs = inputs.to(env.DEVICE)
                labels = labels.to(env.DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase is schemas.constants.Phase.TRAINING):
                    outputs = model(inputs.float())
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase is schemas.constants.Phase.TRAINING:
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data).double()

            epoch_loss = epoch_loss / len(dataloaders[phase.value].dataset)
            epoch_acc = epoch_corrects / len(dataloaders[phase.value].dataset)

            if scheduler and phase is schemas.constants.Phase.TRAINING:
                scheduler.step()
                local_logger.info("Last learning rate in this epoch: %.3f", scheduler.get_last_lr()[0])

            local_logger.info("%s Loss: %.4f Acc: %.4f.", phase, epoch_loss, epoch_acc)

            if phase is schemas.constants.Phase.VALIDATION:
                if best_criteria is schemas.constants.BestCriteria.LOSS and epoch_loss < best_record:
                    local_logger.info("New Record: %.4f < %.4f", epoch_loss, best_record)
                    best_record = epoch_loss
                    best_weights = deepcopy(model.state_dict())
                    local_logger.debug("Updated best models.")

                if best_criteria is schemas.constants.BestCriteria.ACCURACY and epoch_acc > best_record:
                    local_logger.info("New Record: %.4f < %.4f", epoch_acc, best_record)
                    best_record = epoch_acc
                    best_weights = deepcopy(model.state_dict())
                    local_logger.debug("Updated best models.")

            train_acc[phase.value].append(float(epoch_acc))
            train_loss[phase.value].append(float(epoch_loss))
            local_logger.debug("Updated train_acc, train_loss: %.4f, %.4f", epoch_acc, epoch_loss)

    last_weights = deepcopy(model.state_dict())
    time_elapsed = (datetime.now() - training_start).total_seconds()
    local_logger.info("Training complete at %s", datetime.now())
    local_logger.info("Training complete in %dm %ds.", time_elapsed // 60, time_elapsed % 60)
    local_logger.info("Best val %s: %.4f}.", best_criteria, best_record)
    return model, best_weights, last_weights, train_loss, train_acc
