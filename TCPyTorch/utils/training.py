import os
import torch

from datetime import datetime
from copy import deepcopy
from tqdm import tqdm

from .common import save_dict_as_yml
from .logger import get_logger

DEVICE = os.getenv("DEVICE")
LOGGER = get_logger("Training")
AVAILABLE_OPTIMIZER = ["sgd", "rmsprop", "adam", "adamw"]
AVAILABLE_SCHEDULER = ["step", "cosine"]
AVAILABLE_STANDARD = ["loss", "acc"]


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs: int, standard: str):
    if standard not in AVAILABLE_STANDARD:
        raise NotImplementedError(f"Training with {standard=} is invalid or not implemented.")

    training_start = datetime.now()
    LOGGER.info("Start time of training: %s", training_start)
    LOGGER.info("Training using device: %s", DEVICE)
    model.to(DEVICE)
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
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

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
                LOGGER.info("Last learning rate in this epoch: %.3f", scheduler.get_last_lr()[0])
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
            LOGGER.debug("Updated train_acc, train_loss: %.4f, %.4f", epoch_acc, epoch_loss)

    last_weights = deepcopy(model.state_dict())
    training_end = datetime.now()
    time_elapsed = (training_end - training_start).total_seconds()
    LOGGER.info("Training complete at %s", training_end)
    LOGGER.info("Training complete in %dm %ds.", time_elapsed // 60, time_elapsed % 60)
    LOGGER.info("Best val %s: %.4f}.", standard, best_record)
    return model, best_weights, last_weights, train_loss, train_acc


def get_optimizer(params, name="adam", lr=1e-3, momentum=0.9, weight_decay=0.0, alpha=0.99, betas=(0.9, 0.999)):
    if name.lower() not in AVAILABLE_OPTIMIZER:
        raise NotImplementedError("The optimizer is not implemented in this TAO-like pytorch classifier.")

    LOGGER.info("Creating optimizer: %s", name)
    if name.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name.lower() == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay, alpha=alpha)
    elif name.lower() == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif name.lower() == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)


def get_scheduler(name: str, optimizer: torch.optim, num_epochs: int, step_size=30, gamma=0.1, lr_min=0):
    if name.lower() not in AVAILABLE_SCHEDULER:
        raise NotImplementedError("The scheduler is not implemented in this TAO-like pytorch classifier.")

    LOGGER.info("Creating scheduler: %s", name)
    if name.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    elif name.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=lr_min)


def get_class_mapping(dataset, savepath: str = None) -> dict:
    mapping = dataset.class_to_idx
    LOGGER.info("Reading class mapping in the dataset: %s", mapping)
    if savepath:
        save_dict_as_yml(savepath, mapping)
    return mapping
