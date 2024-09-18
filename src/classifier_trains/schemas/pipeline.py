from typing import Optional

from pydantic import BaseModel, ValidationInfo, field_validator, model_validator

import classifier_trains.schemas.config as C
from classifier_trains.utils.logger import get_logger


local_logger = get_logger(__name__)


class PipelineConfig(BaseModel):
    """Pipeline configuration"""

    enable_training: bool
    enable_evaluation: bool
    preprocessing: C.PreprocessingConfig
    dataloader: C.DataloaderConfig
    model: Optional[C.ModelConfig] = None
    training: Optional[C.TrainingConfig] = None
    evaluation: Optional[C.EvaluationConfig] = None

    @field_validator("training")
    @classmethod
    def training_is_required_if_enable_training(
        cls, v: Optional[C.TrainingConfig], info: ValidationInfo
    ) -> Optional[C.TrainingConfig]:
        """Ensure training is required if enabled"""

        if info.data["enable_training"] and not v:
            raise ValueError("Training config is required if enable training")

        if not info.data["enable_training"]:
            if v:
                local_logger.warning("Training config is provided but ignored because training is disabled.")
            return None

        return v

    @field_validator("model")
    @classmethod
    def model_is_required_if_enable_training(
        cls, v: Optional[C.TrainingConfig], info: ValidationInfo
    ) -> Optional[C.TrainingConfig]:
        """Ensure training is required if enabled"""

        if info.data["enable_training"] and (not v):
            raise ValueError("Model config is required if enable training")

        if not info.data["enable_training"]:
            if v:
                local_logger.warning("Model config is provided but ignored because training is disabled.")
            return None

        return v

    @field_validator("evaluation")
    @classmethod
    def evaluation_is_required_if_enable_evaluation(
        cls, v: Optional[C.EvaluationConfig], info: ValidationInfo
    ) -> Optional[C.EvaluationConfig]:
        """Ensure evaluation is required if enabled"""

        if info.data["enable_evaluation"] and not v:
            raise ValueError("evaluation is required if enabled")

        if not info.data["enable_evaluation"]:
            return None

        return v

    @model_validator(mode="after")
    @classmethod
    def consistent_random_seed_in_training_and_evaluation(cls, v):
        """Ensure the random seed is consistent in training and evaluation"""

        if not (v.enable_training and v.enable_evaluation):
            return v

        if v.training.random_seed != v.evaluation.random_seed:
            raise ValueError("Random seed in training and evaluation must be the same")

        return v
