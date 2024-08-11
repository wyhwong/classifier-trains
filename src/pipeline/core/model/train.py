from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torchvision
from tqdm import tqdm

import pipeline.env
import pipeline.logger
import pipeline.schemas.constants


local_logger = pipeline.logger.get_logger(__name__)


def train_model(
    model: torchvision.models,
    dataloaders: dict[pipeline.schemas.constants.Phase, torchvision.datasets.ImageFolder],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int,
    best_criteria: pipeline.schemas.constants.BestCriteria,
):
    """
    Train the model.

    Args:
    -----
        model: torchvision.models
            Model to train.

        dataloaders: dict[pipeline.schemas.constants.Phase, torchvision.datasets.ImageFolder]
            Dataloaders for training and validation.

        criterion: torch.nn.Module
            Criterion for the model.

        optimizer: torch.optim
            Optimizer for the model.

        scheduler: torch.optim.lr_scheduler
            Scheduler for the optimizer.

        num_epochs: int
            Number of epochs to train.

        best_criteria: pipeline.schemas.constants.BestCriteria
            Best criteria to save the best model.

    Returns:
    --------
        model: torchvision.models
            Trained model.

        best_weights: dict
            Best weights of the model.

        last_weights: dict
            Last weights of the model.

        loss: dict[pipeline.schemas.constants.Phase, list]
            Training loss of the model.

        accuracy: dict[pipeline.schemas.constants.Phase, list]
            Training accuracy of the model.
    """

    training_start = datetime.now()
    local_logger.info("Start time of training: %s", training_start)
    local_logger.info("Training using device: %s", pipeline.envDEVICE)

    # Initialize the best weights and the loss/accuracy record
    best_weights = deepcopy(model.state_dict())
    loss: dict[pipeline.schemas.constants.Phase, list[float]] = {
        pipeline.schemas.constants.Phase.TRAINING: [],
        pipeline.schemas.constants.Phase.VALIDATION: [],
    }
    accuracy: dict[pipeline.schemas.constants.Phase, list[float]] = {
        pipeline.schemas.constants.Phase.TRAINING: [],
        pipeline.schemas.constants.Phase.VALIDATION: [],
    }
    best_record = np.inf if best_criteria is pipeline.schemas.constants.BestCriteria.LOSS else -np.inf

    # Start training
    for epoch in range(1, num_epochs + 1):
        local_logger.info("-" * 40)
        local_logger.info("Epoch %d/%d", epoch, num_epochs)
        local_logger.info("-" * 20)

        for phase in [pipeline.schemas.constants.Phase.TRAINING, pipeline.schemas.constants.Phase.VALIDATION]:
            if phase is pipeline.schemas.constants.Phase.TRAINING:
                local_logger.debug("The %d-th epoch training started.", epoch)
                model.train()
            else:
                local_logger.debug("The %d-th epoch validation started.", epoch)
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(pipeline.envDEVICE)
                labels = labels.to(pipeline.envDEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase is pipeline.schemas.constants.Phase.TRAINING):
                    outputs = model(inputs.float())
                    prediction_loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase is pipeline.schemas.constants.Phase.TRAINING:
                        prediction_loss.backward()
                        optimizer.step()

                epoch_loss += prediction_loss.item() * inputs.size(0)
                epoch_corrects += int(torch.sum(preds == labels.data))

            epoch_loss = epoch_loss / len(dataloaders[phase].dataset)
            epoch_acc = epoch_corrects / len(dataloaders[phase].dataset)

            if scheduler and phase is pipeline.schemas.constants.Phase.TRAINING:
                scheduler.step()
                local_logger.info("Last learning rate in this epoch: %.3f", scheduler.get_last_lr()[0])

            local_logger.info("%s Loss: %.4f Acc: %.4f.", phase, epoch_loss, epoch_acc)

            if phase is pipeline.schemas.constants.Phase.VALIDATION:
                if best_criteria is pipeline.schemas.constants.BestCriteria.LOSS and epoch_loss < best_record:
                    local_logger.info("New Record: %.4f < %.4f", epoch_loss, best_record)
                    best_record = epoch_loss
                    best_weights = deepcopy(model.state_dict())
                    local_logger.debug("Updated best models.")

                if best_criteria is pipeline.schemas.constants.BestCriteria.ACCURACY and epoch_acc > best_record:
                    local_logger.info("New Record: %.4f < %.4f", epoch_acc, best_record)
                    best_record = epoch_acc
                    best_weights = deepcopy(model.state_dict())
                    local_logger.debug("Updated best models.")

            accuracy[phase].append(float(epoch_acc))
            loss[phase].append(float(epoch_loss))

            local_logger.debug(
                "Updated %s accuracy: %.4f, loss: %.4f",
                phase,
                epoch_acc,
                epoch_loss,
            )

    last_weights = deepcopy(model.state_dict())
    time_elapsed = (datetime.now() - training_start).total_seconds()
    local_logger.info("Training complete at %s", datetime.now())
    local_logger.info("Training complete in %dm %ds.", time_elapsed // 60, time_elapsed % 60)
    local_logger.info("Best val %s: %.4f}.", best_criteria, best_record)
    return (model, best_weights, last_weights, loss, accuracy)
