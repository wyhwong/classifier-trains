from typing import Optional

from pipeline.core.preprocessing import Preprocessor
from pipeline.schemas.config import (
    DataloaderConfig,
    EvaluationConfig,
    ExportConfig,
    ModelConfig,
    PreprocessingConfig,
    TrainingConfig,
)


class ClassifierFacade:
    """Class to handle the classifier training"""

    def __init__(
        self,
        preprocessing_config: PreprocessingConfig,
        model_config: ModelConfig,
        training_config: Optional[TrainingConfig],
        dataloader_config: Optional[DataloaderConfig],
        evalution_config: Optional[EvaluationConfig],
        export_config: Optional[ExportConfig],
    ):
        """Initialize the classifier facade

        Args:
            preprocessing_config (PreprocessingConfig): The preprocessing configuration
            model_config (ModelConfig): The model configuration
            training_config (Optional[TrainingConfig]): The training configuration
            dataloader_config (Optional[DataloaderConfig]): The dataloader configuration
            evalution_config (Optional[EvaluationConfig]): The evaluation configuration
            export_config (Optional[ExportConfig]): The export configuration
        """

        self.__preprocessor = Preprocessor(preprocessing_config)
