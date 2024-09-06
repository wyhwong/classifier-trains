from pydantic import BaseModel, model_validator

import classifier_trains.schemas.config as C


class PipelineConfig(BaseModel):
    """Pipeline configuration"""

    enable_training: bool
    enable_evaluation: bool
    model: C.ModelConfig
    dataloader: C.DataloaderConfig
    preprocessing: C.PreprocessingConfig
    training: C.TrainingConfig
    evaluation: C.EvaluationConfig

    @model_validator(mode="after")
    def consistent_random_seed_in_training_and_evaluation(self) -> "PipelineConfig":
        """Ensure the random seed is consistent in training and evaluation"""

        if not (self.enable_training and self.enable_evaluation):
            return self

        if self.training.random_seed != self.evaluation.random_seed:
            raise ValueError("Random seed in training and evaluation must be the same")

        return self
