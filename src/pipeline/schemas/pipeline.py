from pydantic import BaseModel

import pipeline.schemas.config as C


class PipelineConfig(BaseModel):
    """Pipeline configuration"""

    experiment_label: str
    random_seed: int
    device: str
    enable_evaluation: bool
    model: C.ModelConfig
    dataloader: C.DataloaderConfig
    preprocessing: C.PreprocessingConfig
    training: C.TrainingConfig
    evaluation: C.EvaluationConfig
