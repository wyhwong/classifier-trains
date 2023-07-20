import torch
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm

from .common import save_dict_as_yml
from .logger import get_logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOGGER = get_logger("Training")
AVAILABLE_OPTIMIZER = ["sgd", "rmsprop", "adam", "adamw"]
AVAILABLE_SCHEDULER = ["step", "cosine"]
AVAILABLE_STANDARD = ["loss", "acc"]


def trainModel(model, dataloaders, criterion, optimizer, scheduler, num_epochs: int, standard: str):
    if standard not in AVAILABLE_STANDARD:
        raise NotImplementedError(f"Training with {standard=} is invalid or not implemented.")
    training_start = datetime.now()
    LOGGER.info(f"Start time of training {training_start}.")
    LOGGER.info(f"Training using device: {DEVICE}")
    model.to(DEVICE)
    best_weights = deepcopy(model.state_dict())
    train_loss = {"train": [], "val": []}
    train_acc = {"train": [], "val": []}
    best_record = None

    for epoch in range(1, num_epochs + 1):
        LOGGER.info("-" * 40)
        LOGGER.info(f"Epoch {epoch}/{num_epochs}")
        LOGGER.info("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                LOGGER.debug(f"The {epoch}-th epoch training started.")
                model.train()
            else:
                LOGGER.debug(f"The {epoch}-th epoch validation started.")
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
                LOGGER.info(f"Last learning rate in this epoch: {scheduler.get_last_lr()}.")
            LOGGER.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}.")

            if phase == "val" and best_record is None:
                if standard == "loss":
                    best_record = epoch_loss
                if standard == "acc":
                    best_record = epoch_acc

            if phase == "val":
                if standard == "loss" and epoch_loss < best_record:
                    LOGGER.info(f"New Record: {epoch_loss=} < {best_record}.")
                    best_record = epoch_loss
                    best_weights = deepcopy(model.state_dict())
                    LOGGER.debug("Updated best models.")
                if standard == "acc" and epoch_acc > best_record:
                    LOGGER.info(f"New Record: {epoch_acc=} > {best_record}.")
                    best_record = epoch_acc
                    best_weights = deepcopy(model.state_dict())
                    LOGGER.debug("Updated best models.")

            train_acc[phase].append(float(epoch_acc))
            train_loss[phase].append(float(epoch_loss))
            LOGGER.debug(f"Updated {train_acc=}, {train_loss=}.")

    last_weights = deepcopy(model.state_dict())
    training_end = datetime.now()
    time_elapsed = (training_end - training_start).total_seconds()
    LOGGER.info(f"Training complete at {training_end}")
    LOGGER.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.")
    LOGGER.info(f"Best val {standard}: {best_record:4f}.")
    return model, best_weights, last_weights, train_loss, train_acc


def get_optimizer(params, name="adam", lr=1e-3, momentum=0.9, weight_decay=0, alpha=0.99, betas=(0.9, 0.999)):
    if name.lower() not in AVAILABLE_OPTIMIZER:
        raise NotImplementedError("The optimizer is not implemented in this TAO-like pytorch classifier.")
    if name.lower() == "sgd":
        LOGGER.debug(f"Creating optimizer {name}, {lr=}, {momentum=}, {weight_decay=}.")
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name.lower() == "rmsprop":
        LOGGER.debug(f"Creating optimizer {name}, {lr=}, {momentum=}, {weight_decay=}, {alpha=}.")
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay, alpha=alpha)
    elif name.lower() == "adam":
        LOGGER.debug(f"Creating optimizer {name}, {betas=}, {weight_decay=}.")
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif name.lower() == "adamw":
        LOGGER.debug(f"Creating optimizer {name}, {betas=}, {weight_decay=}.")
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)


def get_scheduler(name: str, optimizer: torch.optim, num_epochs: int, step_size=30, gamma=0.1, lr_min=0):
    if name.lower() not in AVAILABLE_SCHEDULER:
        raise NotImplementedError("The scheduler is not implemented in this TAO-like pytorch classifier.")
    if name.lower() == "step":
        LOGGER.debug(f"Creating scheduler {name}, {step_size=}, {gamma=}.")
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    elif name.lower() == "cosine":
        LOGGER.debug(f"Creating scheduler {name}, {num_epochs=}, {lr_min=}")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=lr_min)


def get_class_mapping(dataset, savepath=None, save=False) -> dict:
    mapping = dataset.class_to_idx
    LOGGER.info(f"Reading class mapping in the dataset: {mapping}")
    if save:
        if savepath is None:
            raise ValueError("The savepath cannot be None if save=True")
        save_dict_as_yml(ymlpath=savepath, input_dict=mapping)
    return mapping
