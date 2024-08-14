import datetime

import torch
from torch import nn

import pipeline.logger
import pipeline.core.model.utils
from pipeline.schemas import config, constants

local_logger = pipeline.logger.get_logger(__name__)


class ModelFacade:
    """Class to handle the model related functions.

    TODO: Migrate to pytorch lightning
    """

    def __init__(self, model_config: config.ModelConfig) -> None:
        """Initialize the ModelFacade object.

        Args:
            model_config (config.ModelConfig): The model configuration
        """

        local_logger.info("Initializing ModelFacade with config %s", model_config)

        self.__model_config = model_config
        self.__model = pipeline.core.model.utils.initialize_model(
            model_config=self.__model_config,
        )

    def train(
        self,
        num_epochs: int,
        best_criteria: constants.BestCriteria,
        optimizer_config: config.OptimizerConfig,
        scheduler_config: config.SchedulerConfig,
    ) -> None:
        """Train the model.

        Args:
            num_epochs (int): The number of epochs to train the model
            best_criteria (constants.BestCriteria): The criteria to select the best model
            optimizer_config (config.OptimizerConfig): The optimizer configuration
            scheduler_config (config.SchedulerConfig): The scheduler configuration
        """

        training_start = datetime.datetime.now(tz=datetime.timezone.utc)
        local_logger.info("Training started at %s.", training_start)

        optimizer = pipeline.core.model.utils.initialize_optimizer(
            params=self.__model.parameters(),
            optimizer_config=optimizer_config,
        )
        scheduler = pipeline.core.model.utils.initialize_scheduler(
            optimizer=optimizer,
            scheduler_config=scheduler_config,
            num_epochs=num_epochs,
        )

    def inference(self, data: torch.Tensor) -> torch.Tensor:
        """Perform inference"""

    def export(self, export_path: str, is_weights_only: bool = False) -> None:
        """Export the model

        Args:
            export_path (str): The path to export the model
            is_weights_only (bool, optional): Whether to export only the weights. Defaults
        """
