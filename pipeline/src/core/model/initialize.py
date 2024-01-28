from typing import Any, Iterator

import torch
import torchvision

import env
import logger
import schemas.constants


local_logger = logger.get_logger(__name__)


def initialize_model(
    backbone: schemas.constants.ModelBackbone,
    weights: str,
    num_classes: int,
    unfreeze_all_params: bool,
) -> torchvision.models:
    """
    Initializes a model with the specified backbone, weights, and number of classes.

    Args:
    -----
        backbone (schemas.constants.ModelBackbone):
            The backbone architecture of the model.

        weights (str):
            The path to the pre-trained weights file.

        num_classes (int):
            The number of classes in the dataset.

        unfreeze_all_params (bool):
            Whether to unfreeze all parameters of the model.

    Returns:
    -----
        model (torchvision.models):
            The initialized model.
    """

    local_logger.info("Initializing model backbone: %s.", backbone.value)
    local_logger.debug(
        "The path to the pre-trained weights file: %s, the number of classes: %d, unfreeze all parameters: %s.",
        weights,
        num_classes,
        unfreeze_all_params,
    )

    model = getattr(torchvision.models, backbone.value)(weights=weights)
    # Modify output layer to fit number of classes
    if "resnet" in backbone.value:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    if "alexnet" in backbone.value:
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

    if "vgg" in backbone.value:
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

    if "squeezenet" in backbone.value:
        model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes

    if "densenet" in backbone.value:
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    if "inception" in backbone.value:
        model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    local_logger.info("Modified the output layer of the model.")

    if unfreeze_all_params:
        unfreeze_all_params_in_model(model)

    # To enable PyTorch 2 compiler for optimized performance
    # (Only support CUDA capability >= 7.0)
    if env.DEVICE == "cuda" and torch.cuda.get_device_properties("cuda").major >= 7:
        local_logger.info("Enabled PyTorch 2 compiler for optimized performance.")
        model = torch.compile(model)

    # Move the model to the device
    model.to(env.DEVICE)
    local_logger.info("Moved the model to %s.", env.DEVICE)

    return model


def unfreeze_all_params_in_model(model: torchvision.models) -> None:
    """
    Unfreezes all parameters in the model by setting `requires_grad` to True for each parameter.

    Args:
    -----
        model (torchvision.models):
            The model whose parameters need to be unfrozen.

    Returns:
    -----
        None
    """

    for param in model.parameters():
        param.requires_grad = True

    local_logger.info("Unfreezed all parameters in the model.")


def initialize_optimizer(
    params: Iterator[Any],
    optimizier=schemas.constants.OptimizerType,
    lr=1e-3,
    momentum=0.9,
    weight_decay=0.0,
    alpha=0.99,
    betas=(0.9, 0.999),
) -> torch.optim.Optimizer:
    """
    Get optimizer for the model.

    Args:
    -----
        params: Iterator[Any] (torch.nn.Module.parameters)
            Parameters of the model.

        optimizier: schemas.constants.OptimizerType
            Optimizer type.

        lr: float, optional
            Learning rate for the optimizer.

        momentum: float, optional
            Momentum for the optimizer.

        weight_decay: float, optional
            Weight decay for the optimizer.

        alpha: float, optional
            Alpha for the optimizer.

        betas: tuple, optional
            Betas for the optimizer.

    Returns:
    --------
        optimizer: torch.optim.Optimizer
            Optimizer for the model.
    """

    local_logger.info("Creating optimizer: %s.", optimizier.value)
    local_logger.debug(
        "The learning rate: %.4f, momentum: %.4f, weight_decay: %.4f, alpha: %.4f, betas: %s.",
        lr,
        momentum,
        weight_decay,
        alpha,
        betas,
    )

    if optimizier is schemas.constants.OptimizerType.SGD:
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    if optimizier is schemas.constants.OptimizerType.RMSPROP:
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay, alpha=alpha)

    if optimizier is schemas.constants.OptimizerType.ADAM:
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)

    if optimizier is schemas.constants.OptimizerType.ADAMW:
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)

    raise ValueError(f"Invalid optimizer type: {optimizier.value}")


def initialize_scheduler(
    scheduler: schemas.constants.SchedulerType,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    step_size: int = 30,
    gamma: float = 0.1,
    lr_min: float = 0.0,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Get scheduler for the optimizer.

    Args:
    -----
        scheduler: schemas.constants.SchedulerType
            Scheduler type.

        optimizer: torch.optim.Optimizer
            Optimizer to apply scheduler.

        num_epochs: int
            Number of epochs to train.

        step_size: int, optional
            Step size for the scheduler.

        gamma: float, optional
            Gamma for the scheduler.

        lr_min: float, optional
            Minimum learning rate for the scheduler.

    Returns:
    --------
        scheduler: torch.optim.lr_scheduler.LRScheduler
            Scheduler for the optimizer.
    """

    local_logger.info("Creating scheduler: %s for optimizer.", scheduler.value)
    local_logger.debug(
        "The number of epochs: %d, step_size: %d, gamma: %.4f, lr_min: %.4f.",
        num_epochs,
        step_size,
        gamma,
        lr_min,
    )

    if scheduler is schemas.constants.SchedulerType.STEP:
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    if scheduler is schemas.constants.SchedulerType.COSINE:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=lr_min)

    raise ValueError(f"Invalid scheduler type: {scheduler.value}")
