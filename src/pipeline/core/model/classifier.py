from typing import Optional

import lightning as pl
import numpy as np
import torch
from torch import nn
from torchmetrics import Accuracy

import pipeline.core.model.utils
import pipeline.logger
from pipeline.schemas import config, constants


local_logger = pipeline.logger.get_logger(__name__)


class ClassifierModel(pl.LightningModule):
    """Classifier model class"""

    def __init__(self, model_config: config.ModelConfig) -> None:
        """Initialize the ClassifierModel object

        Args:
            model_config (config.ModelConfig): The model configuration
        """

        self.__model_config = model_config

        self.__classifier = pipeline.core.model.utils.initialize_classifier(model_config=model_config)

        if self.__model_config.weights:
            self.__load_weights(weights_path=self.__model_config.weights_path)

        self.__loss_fn: Optional[nn.Module] = None
        self.__optimizers: Optional[torch.optim.Optimizer] = None
        self.__schedulers: Optional[torch.optim.lr_scheduler._LRScheduler] = None

        self.__batch_acc: dict[str, nn.Module] = {
            phase: Accuracy(
                task="multiclass",
                num_classes=self.__model_config.num_classes,
            )
            for phase in constants.Phase
        }
        self.__batch_loss: dict[str, list[float]] = {phase: [] for phase in constants.Phase}
        self.__best_loss: dict[str, float] = {phase: np.inf for phase in constants.Phase}
        self.__best_acc: dict[str, float] = {phase: -np.inf for phase in constants.Phase}

    def __load_weights(self, weights_path: str) -> None:
        """Load the weights from the given path.

        Args:
            weights_path (str): The path to the weights file.
        """

        weights = torch.load(weights_path)
        self.__classifier.load_state_dict(weights)
        local_logger.info("Loaded weights from local file: %s.", weights_path)

    def training_setup(
        self,
        num_epochs: int,
        optimizer_config: config.OptimizerConfig,
        scheduler_config: config.SchedulerConfig,
    ) -> None:
        """Setup the training configuration

        Args:
            num_epochs (int): The number of epochs
            optimizer_config (config.OptimizerConfig): The optimizer configuration
            scheduler_config (config.SchedulerConfig): The scheduler configuration
        """

        self.__optimizers = [
            pipeline.core.model.utils.initialize_optimizer(
                params=self.__classifier.parameters(),
                optimizer_config=optimizer_config,
            )
        ]
        self.__schedulers = [
            pipeline.core.model.utils.initialize_scheduler(
                optimizer=self.__optimizer,
                scheduler_config=scheduler_config,
                num_epochs=num_epochs,
            )
        ]

    def configure_optimizers(self):
        """Configure the optimizers and schedulers."""

        return tuple(
            [
                {
                    "optimizer": optim,
                    "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
                }
                for optim, sched in zip(self.__optimizers, self.__schedulers)
            ]
        )

    def __common_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,  # pylint: disable=unused-argument
        phase: constants.Phase,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        x, y = batch
        logits = self.__classifier(x)
        loss = self.__loss_fn(logits, y)
        self.log(phase("loss"), loss, on_step=True, on_epoch=False)

        acc = self.__batch_acc[phase](logits, y)
        self.log(phase("accuracy"), acc, on_step=True, on_epoch=False)

        return loss

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The input batch
            batch_idx (int): The batch index

        Returns:
            torch.Tensor: The loss
        """

        loss = self.__common_step(
            batch=batch,
            batch_idx=batch_idx,
            phase=constants.Phase.TRAIN,
        )
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Validation step

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The input batch
            batch_idx (int): The batch index

        Returns:
            torch.Tensor: The loss
        """

        loss = self.__common_step(
            batch=batch,
            batch_idx=batch_idx,
            phase=constants.Phase.VALID,
        )
        return loss

    def __common_epoch_end(self, phase: constants.Phase) -> None:

        epoch_loss = sum(self.__batch_loss[phase]) / len(self.__batch_loss[phase])
        epoch_acc = sum(self.__batch_acc[phase]) / len(self.__batch_acc[phase])

        self.__batch_loss[phase].clear()
        self.__batch_r2[phase].clear()

        self.__best_acc[phase] = max(epoch_acc, self.__best_acc[phase])
        self.__best_loss[phase] = min(epoch_loss, self.__best_loss[phase])

        self.log(phase("epoch_loss"), epoch_loss, on_step=False, on_epoch=True)
        self.log(phase("epoch_accuracy"), epoch_acc, on_step=False, on_epoch=True)

    def training_epoch_end(self) -> None:
        """Training epoch end"""

        self.__common_epoch_end(phase=constants.Phase.TRAIN)

    def validation_epoch_end(self) -> None:
        """Validation epoch end"""

        self.__common_epoch_end(phase=constants.Phase.VALID)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """

        return self.__classifier(x)
