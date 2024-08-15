from pydantic import BaseModel

import pipeline.schemas.config as C


class PipelineConfig(BaseModel):
    """Pipeline configuration"""

    experiment_label: str
    enable_training: bool
    enable_evaluation: bool
    enable_export: bool
    model: C.ModelConfig
    dataloader: C.DataloaderConfig
    preprocessing: C.PreprocessingConfig
    training: C.TrainingConfig
    evaluation: C.EvaluationConfig
