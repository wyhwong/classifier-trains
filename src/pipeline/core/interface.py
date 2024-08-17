from typing import Optional

import torch
import torchvision.datasets as datasets

import pipeline.core.utils
import pipeline.logger
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

    def __save_class_mapping(
        dataset: datasets.ImageFolder,
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Get class mapping from the dataset.

        Args:
            dataset (torchvision.datasets.ImageFolder): Dataset to get class mapping.
            output_dir (str, optional): The output directory. Defaults to None.
        """

        mapping = dataset.class_to_idx
        local_logger.info("Reading class mapping in the dataset: %s.", mapping)

        if output_dir:
            savepath = output_dir + "/class_mapping.yml"
            pipeline.core.utils.save_as_yml(savepath, mapping)
            local_logger.info("Saved class mapping to %s.", savepath)

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

        transforms = self.__preprocessor.construct_transforms_compose()
        dataloaders: dict[constants.Phase, torch.utils.data.DataLoader] = {}

        for phase in [constants.Phase.TRAINING, constants.Phase.VALIDATION]:
            image_dataset = datasets.ImageFolder(
                root=dataloader_config.get_dirpath(phase),
                transform=transforms[phase],
            )
            self.__save_class_mapping(image_dataset, output_dir)
            dataloaders[phase] = torch.utils.data.DataLoader(
                image_dataset,
                shuffle=True,
                batch_size=dataloader_config.batch_size,
                num_workers=dataloader_config.num_workers,
            )

        self.__model_facade.train(
            dataloaders=dataloaders,
            num_epochs=training_config.num_epochs,
            best_criteria=training_config.best_criteria,
            optimizer_config=training_config.optimizer,
            scheduler_config=training_config.scheduler,
        )

        training_history = self.__model_facade.get_training_history()

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

        return self.__model_facade.inference(data=data)

    def export(self, output_dir: str) -> None:
        """Export the model

        Args:
            output_dir (str): The output directory
        """

        pipeline.core.utils.check_and_create_dir(output_dir)
