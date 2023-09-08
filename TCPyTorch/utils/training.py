import torch
import torchvision
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm

import utils.common
import utils.logger
import utils.constants
import utils.env

LOGGER = utils.logger.get_logger("Training")


def train_model(
    model: torchvision.models,
    dataloaders: dict[str, torchvision.datasets.ImageFolder],
    criterion: torch.nn.Module,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    num_epochs: int,
    standard: str,
) -> tuple:
    """
    Train the model with the given dataloaders, criterion, optimizer, scheduler, and standard.

    Parameters
    ----------
    model: torch.nn.Module, required, the model to be trained.
    dataloaders: dict, required, the dataloaders for training.
    criterion: torch.nn.Module, required, the criterion for training.
    optimizer: torch.optim, required, the optimizer for training.
    scheduler: torch.optim.lr_scheduler, required, the scheduler for training.
    num_epochs: int, required, the number of epochs for training.
    standard: str, required, the standard for training, either "loss" or "acc".

    Returns
    -------
    model: torch.nn.Module, the trained model.
    best_weights: dict, the best weights of the model.
    last_weights: dict, the last weights of the model.
    train_loss: dict, the training loss of the model.
    train_acc: dict, the training accuracy of the model.
    """
    if standard not in utils.constants.AVAILABLE_STANDARD:
        raise NotImplementedError(
            f"Training with {standard=} is invalid or not implemented."
        )

    training_start = datetime.now()
    LOGGER.info("Start time of training: %s", training_start)
    LOGGER.info("Training using device: %s", utils.env.DEVICE)
    model.to(utils.env.DEVICE)
    best_weights = deepcopy(model.state_dict())
    train_loss = {"train": [], "val": []}
    train_acc = {"train": [], "val": []}
    best_record = None

    for epoch in range(1, num_epochs + 1):
        LOGGER.info("-" * 40)
        LOGGER.info("Epoch %d/%d", epoch, num_epochs)
        LOGGER.info("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                LOGGER.debug("The %d-th epoch training started.", epoch)
                model.train()
            else:
                LOGGER.debug("The %d-th epoch validation started.", epoch)
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(utils.env.DEVICE)
                labels = labels.to(utils.env.DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs.float())
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders[phase].dataset)

            if scheduler and phase == "train":
                scheduler.step()
                LOGGER.info(
                    "Last learning rate in this epoch: %.3f", scheduler.get_last_lr()[0]
                )
            LOGGER.info("%s Loss: %.4f Acc: %.4f.", phase, epoch_loss, epoch_acc)

            if phase == "val" and best_record is None:
                if standard == "loss":
                    best_record = epoch_loss
                if standard == "acc":
                    best_record = epoch_acc

            if phase == "val":
                if standard == "loss" and epoch_loss < best_record:
                    LOGGER.info("New Record: %.4f < %.4f", epoch_loss, best_record)
                    best_record = epoch_loss
                    best_weights = deepcopy(model.state_dict())
                    LOGGER.debug("Updated best models.")
                if standard == "acc" and epoch_acc > best_record:
                    LOGGER.info("New Record: %.4f < %.4f", epoch_acc, best_record)
                    best_record = epoch_acc
                    best_weights = deepcopy(model.state_dict())
                    LOGGER.debug("Updated best models.")

            train_acc[phase].append(float(epoch_acc))
            train_loss[phase].append(float(epoch_loss))
            LOGGER.debug(
                "Updated train_acc, train_loss: %.4f, %.4f", epoch_acc, epoch_loss
            )

    last_weights = deepcopy(model.state_dict())
    training_end = datetime.now()
    time_elapsed = (training_end - training_start).total_seconds()
    LOGGER.info("Training complete at %s", training_end)
    LOGGER.info("Training complete in %dm %ds.", time_elapsed // 60, time_elapsed % 60)
    LOGGER.info("Best val %s: %.4f}.", standard, best_record)
    return model, best_weights, last_weights, train_loss, train_acc


def get_optimizer(
    params: torch.nn.Module.parameters,
    name="adam",
    lr=1e-3,
    momentum=0.9,
    weight_decay=0.0,
    alpha=0.99,
    betas=(0.9, 0.999),
) -> torch.optim:
    """
    Get the optimizer with the given parameters.

    Parameters
    ----------
    params: torch.nn.Module.parameters, required, the parameters of the model.
    name: str, optional, the name of the optimizer.
    lr: float, optional, the learning rate of the optimizer.
    momentum: float, optional, the momentum of the optimizer (if applicable).
    weight_decay: float, optional, the weight decay of the optimizer (if applicable).
    alpha: float, optional, the alpha of the optimizer (if applicable).
    betas: tuple, optional, the betas of the optimizer (if applicable).

    Returns
    -------
    optimizer: torch.optim, the optimizer.
    """
    if name.lower() not in utils.constants.AVAILABLE_OPTIMIZER:
        raise NotImplementedError(
            "The optimizer is not implemented in this TAO-like pytorch classifier."
        )

    LOGGER.info("Creating optimizer: %s", name)
    if name.lower() == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif name.lower() == "rmsprop":
        return torch.optim.RMSprop(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay, alpha=alpha
        )
    elif name.lower() == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif name.lower() == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)


def get_scheduler(
    name: str,
    optimizer: torch.optim,
    num_epochs: int,
    step_size=30,
    gamma=0.1,
    lr_min=0,
) -> torch.optim.lr_scheduler:
    """
    Get the scheduler with the given parameters.

    Parameters
    ----------
    name: str, required, the name of the scheduler.
    optimizer: torch.optim, required, the optimizer.
    num_epochs: int, required, the number of epochs (if applicable).
    step_size: int, optional, the step size of the scheduler (if applicable).
    gamma: float, optional, the gamma of the scheduler (if applicable).
    lr_min: float, optional, the minimum learning rate of the scheduler (if applicable).

    Returns
    -------
    scheduler: torch.optim.lr_scheduler, the scheduler.
    """
    if name.lower() not in utils.constants.AVAILABLE_SCHEDULER:
        raise NotImplementedError(
            "The scheduler is not implemented in this TAO-like PyTorch classifier."
        )

    LOGGER.info("Creating scheduler: %s", name)
    if name.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=step_size, gamma=gamma
        )
    elif name.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=num_epochs, eta_min=lr_min
        )


def get_class_mapping(
    dataset: torchvision.datasets.ImageFolder, savepath: str = None
) -> dict:
    """
    Get the class mapping in the dataset.

    Parameters
    ----------
    dataset: torchvision.datasets.ImageFolder, required, the dataset.
    savepath: str, optional, the savepath of the class mapping.

    Returns
    -------
    mapping: dict, the class mapping.
    """
    mapping = dataset.class_to_idx
    LOGGER.info("Reading class mapping in the dataset: %s", mapping)
    if savepath:
        utils.common.save_dict_as_yml(savepath, mapping)
    return mapping
