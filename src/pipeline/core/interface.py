from typing import Optional

import torch

import pipeline.core.utils
import pipeline.logger
from pipeline.core.dataloader import get_datamodule_for_training
from pipeline.core.model import ModelFacade
from pipeline.core.preprocessing import Preprocessor
from pipeline.core.visualization import Visualizer
from pipeline.schemas import constants
from pipeline.schemas.config import DataloaderConfig, EvaluationConfig, ModelConfig, PreprocessingConfig, TrainingConfig


local_logger = pipeline.logger.get_logger(__name__)


class ModelInterface:
    """Class to handle the classifier training"""

    def __init__(
        self,
        preprocessing_config: PreprocessingConfig,
        model_config: ModelConfig,
    ) -> None:
        """Initialize the classifier facade

        Args:
            preprocessing_config (PreprocessingConfig): The preprocessing configuration
            model_config (ModelConfig): The model configuration
        """

        self.__preprocessor = Preprocessor(preprocessing_config=preprocessing_config)
        self.__model_facade = ModelFacade(model_config=model_config)
        self.__visualizer = Visualizer()

    def train(
        self,
        training_config: TrainingConfig,
        dataloader_config: DataloaderConfig,
        output_dir: Optional[str] = None,
    ) -> None:
        """Train the classifier

        Args:
            training_config (TrainingConfig): The training configuration
            dataloader_config (DataloaderConfig): The dataloader configuration
            output_dir (Optional[str], optional): The output directory. Defaults to None.
        """

        if output_dir:
            pipeline.core.utils.check_and_create_dir(output_dir)

        transforms = {
            constants.Phase.TRAINING: self.__preprocessor.get_training_transforms(),
            constants.Phase.VALIDATION: self.__preprocessor.get_validation_transforms(),
        }
        datamodule = get_datamodule_for_training(
            dataloader_config=dataloader_config,
            transforms=transforms,
        )

        self.__model_facade.train(
            training_config=training_config,
            datamodule=datamodule,
            output_dir=output_dir,
        )

    def evaluate(
        self,
        evaluation_config: EvaluationConfig,
        output_dir: Optional[str] = None,
    ):
        """Evaluate the model

        Args:
            evaluation_config (EvaluationConfig): The evaluation configuration
            output_dir (Optional[str], optional): The output directory. Defaults to None.
        """

        if output_dir:
            pipeline.core.utils.check_and_create_dir(output_dir)

    def inference(self, data: torch.Tensor) -> torch.Tensor:
        """Make inference with the model

        Args:
            data: The data to make inference

        Returns:
            torch.Tensor: The inference result
        """

        return self.__model_facade.inference(x=data)
