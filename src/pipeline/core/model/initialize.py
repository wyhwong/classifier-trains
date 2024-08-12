from typing import Any, Iterator

import torch
import torchvision
from torch import nn

import pipeline.env
import pipeline.logger
from pipeline.schemas import config


local_logger = pipeline.logger.get_logger(__name__)


def initialize_model(model_config: config.ModelConfig) -> nn.Module:
    """
    Initializes a model with the specified backbone, weights, and number of classes.

    Args:
    -----
        model_config (pipeline.schemas.config.ModelConfig):

    Returns:
    -----
        model (nn.Module):
            The initialized model.
    """

    local_logger.info("Initializing model with the following configuration: %s.", model_config)

    model = getattr(torchvision.models, model_config.backbone)(weights=model_config.weights)
    # Modify output layer to fit number of classes
    if "resnet" in model_config.backbone:
        model.fc = torch.nn.Linear(model.fc.in_features, model_config.num_classes)

    if "alexnet" in model_config.backbone:
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, model_config.num_classes)

    if "vgg" in model_config.backbone:
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, model_config.num_classes)

    if "squeezenet" in model_config.backbone:
        model.classifier[1] = torch.nn.Conv2d(512, model_config.num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = model_config.num_classes

    if "densenet" in model_config.backbone:
        model.classifier = torch.nn.Linear(model.classifier.in_features, model_config.num_classes)

    if "inception" in model_config.backbone:
        model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, model_config.num_classes)
        model.fc = torch.nn.Linear(model.fc.in_features, model_config.num_classes)

    local_logger.info("Modified the output layer of the model.")

    if model_config.unfreeze_all_params:
        unfreeze_all_params_in_model(model)

    # To enable PyTorch 2 compiler for optimized performance
    # (Only support CUDA capability >= 7.0)
    if pipeline.env.DEVICE == "cuda" and torch.cuda.get_device_properties("cuda").major >= 7:
        local_logger.info("Enabled PyTorch 2 compiler for optimized performance.")
        model = torch.compile(model)

    # Move the model to the device
    model.to(pipeline.env.DEVICE)
    local_logger.info("Moved the model to %s.", pipeline.env.DEVICE)

    return model


def unfreeze_all_params_in_model(model: nn.Module) -> None:
    """
    Unfreezes all parameters in the model by setting `requires_grad` to True for each parameter.

    Args:
    -----
        model (nn.Module):
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
    optimizer_config: config.OptimizerConfig,
) -> torch.optim.Optimizer:
    """
    Get optimizer for the model.

    Args:
    -----
        params: Iterator[Any] (torch.nn.Module.parameters)
            Parameters of the model.

        optimizer_config: pipeline.schemas.config.OptimizerConfig
            Optimizer configuration.

    Returns:
    --------
        optimizer: torch.optim.Optimizer
            Optimizer for the model.
    """

    local_logger.info("Creating optimizer with config %s", optimizer_config)

    if optimizer_config.optimizier is pipeline.schemas.constants.OptimizerType.SGD:
        return torch.optim.SGD(
            params,
            lr=optimizer_config.lr,
            momentum=optimizer_config.momentum,
            weight_decay=optimizer_config.weight_decay,
        )

    if optimizer_config.optimizier is pipeline.schemas.constants.OptimizerType.RMSPROP:
        return torch.optim.RMSprop(
            params,
            lr=optimizer_config.lr,
            momentum=optimizer_config.momentum,
            weight_decay=optimizer_config.weight_decay,
            alpha=optimizer_config.alpha,
        )

    if optimizer_config.optimizier is pipeline.schemas.constants.OptimizerType.ADAM:
        return torch.optim.Adam(
            params,
            lr=optimizer_config.lr,
            betas=optimizer_config.betas,
            weight_decay=optimizer_config.weight_decay,
        )

    if optimizer_config.optimizier is pipeline.schemas.constants.OptimizerType.ADAMW:
        return torch.optim.AdamW(
            params,
            lr=optimizer_config.lr,
            betas=optimizer_config.betas,
            weight_decay=optimizer_config.weight_decay,
        )

    raise ValueError(f"Invalid optimizer type: {optimizer_config.optimizier}")


def initialize_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: config.SchedulerConfig,
    num_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Get scheduler for the optimizer.

    Args:
    -----
        optimizer: torch.optim.Optimizer
            Optimizer to apply scheduler.

        scheduler_config: pipeline.schemas.config.SchedulerConfig
            Scheduler configuration.

    Returns:
    --------
        scheduler: torch.optim.lr_scheduler.LRScheduler
            Scheduler for the optimizer.
    """

    local_logger.info("Creating scheduler with config %s", scheduler_config)

    if scheduler_config.scheduler is pipeline.schemas.constants.SchedulerType.STEP:
        return torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=scheduler_config.step_size,
            gamma=scheduler_config.gamma,
        )

    if scheduler_config.scheduler is pipeline.schemas.constants.SchedulerType.COSINE:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=num_epochs,
            eta_min=scheduler_config.lr_min,
        )

    raise ValueError(f"Invalid scheduler type: {scheduler_config.scheduler}")
