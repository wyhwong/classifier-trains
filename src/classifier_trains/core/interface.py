from classifier_trains.core.loading import ImageDataloader
from classifier_trains.core.model import ModelFacade
from classifier_trains.core.preprocessing import Preprocessor
from classifier_trains.schemas import constants
from classifier_trains.schemas.config import (
    DataloaderConfig,
    EvaluationConfig,
    ModelConfig,
    PreprocessingConfig,
    TrainingConfig,
)
from classifier_trains.utils import file, logger


local_logger = logger.get_logger(__name__)


class ModelInterface:
    """Class to handle the classifier training"""

    def __init__(self, preprocessing_config: PreprocessingConfig) -> None:
        """Initialize the classifier facade

        Args:
            preprocessing_config (PreprocessingConfig): The preprocessing configuration
        """

        self._preprocessor = Preprocessor(preprocessing_config=preprocessing_config)
        self._model_facade = ModelFacade(denorm_fn=self._preprocessor.denormalize)
        self._transforms = {
            constants.Phase.TRAINING: self._preprocessor.get_training_transforms(),
            constants.Phase.VALIDATION: self._preprocessor.get_validation_transforms(),
            constants.Phase.TESTING: self._preprocessor.get_validation_transforms(),
        }

    def train(
        self,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        dataloader_config: DataloaderConfig,
        output_dir: str,
    ) -> None:
        """Train the classifier

        Args:
            training_config (TrainingConfig): The training configuration
            dataloader_config (DataloaderConfig): The dataloader configuration
            output_dir (str): The output directory
        """

        if output_dir:
            file.check_and_create_dir(output_dir)

        datamodule = ImageDataloader(
            dataloader_config=dataloader_config,
            transforms=self._transforms,
        )
        datamodule.setup_for_training(
            trainset_dir=training_config.trainset_dir,
            valset_dir=training_config.valset_dir,
            test_dir=training_config.testset_dir,
        )

        self._model_facade.train(
            model_config=model_config,
            training_config=training_config,
            datamodule=datamodule,
            output_dir=output_dir,
            input_sample=self._preprocessor.get_example_array(),
        )

    def evaluate(
        self,
        evaluation_config: EvaluationConfig,
        dataloader_config: DataloaderConfig,
        output_dir: str,
    ) -> None:
        """Evaluate the model

        Args:
            evaluation_config (EvaluationConfig): The evaluation configuration
            dataloader_config (DataloaderConfig): The dataloader configuration
            output_dir (str): The output directory. Defaults to None.
        """

        if output_dir:
            file.check_and_create_dir(output_dir)

        local_logger.info("Evaluation config: %s", evaluation_config)

        datamodule = ImageDataloader(
            dataloader_config=dataloader_config,
            transforms=self._transforms,
        )
        dataloader = datamodule.get_dataloader(
            dirpath=evaluation_config.evalset_dir,
            is_augmented=False,
        )
        ModelFacade.evaluate(
            evaluation_config=evaluation_config,
            dataloader=dataloader,
            output_dir=output_dir,
        )

    @staticmethod
    def compute_mean_and_std(dirpath: str) -> dict[str, list[float]]:
        """Compute the mean and standard deviation of the dataset.

        Args:
            dirpath (str): The directory path.

        Returns:
            dict[str, list[float]]: The mean and standard deviation.
        """

        return Preprocessor.compute_mean_and_std(dirpath=dirpath)

    @staticmethod
    def get_output_mapping(dirpath: str) -> dict[str, int]:
        """Get the output mapping for the dataset.

        Args:
            dirpath (str): The path to the dataset.

        Returns:
            dict[str, int]: The output mapping.
        """

        return ImageDataloader.get_output_mapping(dirpath=dirpath)
