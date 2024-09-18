from typing import Optional

from pydantic import BaseModel, model_validator, field_validator, ValidationInfo

import classifier_trains.schemas.config as C


class PipelineConfig(BaseModel):
    """Pipeline configuration"""

    enable_training: bool
    enable_evaluation: bool
    model: C.ModelConfig
    dataloader: C.DataloaderConfig
    preprocessing: C.PreprocessingConfig
    training: Optional[C.TrainingConfig] = None
    evaluation: Optional[C.EvaluationConfig] = None

    @field_validator("training")
    @classmethod
    def training_is_required_if_enabled(
        cls, v: Optional[C.TrainingConfig], info: ValidationInfo
    ) -> Optional[C.TrainingConfig]:
        """Ensure training is required if enabled"""

        print(info)

        if info.data["enable_training"] and not v:
            raise ValueError("training is required if enabled")

        if not info.data["enable_training"]:
            return None

        return v

    @field_validator("evaluation")
    @classmethod
    def evaluation_is_required_if_enabled(
        cls, v: Optional[C.EvaluationConfig], info: ValidationInfo
    ) -> Optional[C.EvaluationConfig]:
        """Ensure evaluation is required if enabled"""

        if info.data["enable_evaluation"] and not v:
            raise ValueError("evaluation is required if enabled")

        if not info.data["enable_evaluation"]:
            return None

        return v

    @model_validator(mode="after")
    def consistent_random_seed_in_training_and_evaluation(self) -> "PipelineConfig":
        """Ensure the random seed is consistent in training and evaluation"""

        if not (self.enable_training and self.enable_evaluation):
            return self

        if self.training.random_seed != self.evaluation.random_seed:
            raise ValueError("Random seed in training and evaluation must be the same")

        return self
