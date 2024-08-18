from typing import Optional

import lightning as pl
import torch
from torch import nn

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

        super().__init__()

        self.__model_config = model_config

        self.__classifier = pipeline.core.model.utils.initialize_classifier(model_config=self.__model_config)

        self.__loss_fn = nn.CrossEntropyLoss()
        self.__optimizers: Optional[list[torch.optim.Optimizer]] = None
        self.__schedulers: Optional[list[torch.optim.lr_scheduler.LRScheduler]] = None

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
                optimizer=optim,
                scheduler_config=scheduler_config,
                num_epochs=num_epochs,
            )
            for optim in self.__optimizers
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
    ) -> torch.Tensor:
        """Common step for training, validation, and test

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The input batch
            batch_idx (int): The batch index
            phase (constants.Phase): The phase

        Returns:
            torch.Tensor: The loss
        """

        x, y = batch
        logits = self.__classifier(x)
        loss = self.__loss_fn(logits, y)
        self.log(name=phase(constants.Criterion.LOSS), value=loss, on_step=True)

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(name=phase(constants.Criterion.ACCURACY), value=acc, on_step=True)

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
            phase=constants.Phase.TRAINING,
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
            phase=constants.Phase.VALIDATION,
        )
        return loss

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Test step

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): The input batch
            batch_idx (int): The batch index

        Returns:
            torch.Tensor: The loss
        """

        loss = self.__common_step(
            batch=batch,
            batch_idx=batch_idx,
            phase=constants.Phase.TESTING,
        )
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            torch.Tensor: The output tensor
        """

        return self.__classifier(x)
