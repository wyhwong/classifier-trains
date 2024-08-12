from typing import Optional

from pydantic import BaseModel, NonNegativeFloat, PositiveInt

import pipeline.schemas.constants as C


class ModelConfig(BaseModel):
    """Model configuration"""

    backbone: C.ModelBackbone
    num_classes: PositiveInt
    weights: str
    unfreeze_all_params: bool


class DataloaderConfig(BaseModel):
    """Dataset configuration"""

    trainset_dir: str
    valset_dir: str
    batch_size: PositiveInt
    num_workers: PositiveInt


class ResizeConfig(BaseModel):
    """Resize configuration"""

    width: PositiveInt
    height: PositiveInt
    interpolation: C.InterpolationType
    padding: C.PaddingType
    maintain_aspect_ratio: bool


class SpatialTransformConfig(BaseModel):
    """Spatial transformation configuration"""

    hflip_prob: NonNegativeFloat
    vflip_prob: NonNegativeFloat
    max_rotate_in_degree: NonNegativeFloat
    allow_centor_crop: bool
    allow_random_crop: bool


class ColorTransformConfig(BaseModel):
    """Color transformation configuration"""

    allow_gray_scale: bool
    allow_random_color: bool


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration"""

    mean: list[float]
    std: list[float]
    resize_config: ResizeConfig
    spatial_config: SpatialTransformConfig
    color_config: ColorTransformConfig


class OptimizerConfig(BaseModel):
    """Optimizer configuration"""

    name: C.OptimizerType
    lr: NonNegativeFloat
    weight_decay: NonNegativeFloat
    momentum: Optional[NonNegativeFloat]
    alpha: Optional[NonNegativeFloat]
    betas: Optional[tuple[NonNegativeFloat, NonNegativeFloat]]


class SchedulerConfig(BaseModel):
    """Scheduler configuration"""

    name: C.SchedulerType
    lr_min: NonNegativeFloat
    step_size: PositiveInt
    gamma: NonNegativeFloat


class TrainingConfig(BaseModel):
    """Training configuration"""

    num_epochs: PositiveInt
    best_criteria: C.BestCriteria
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig


class ModelCheckpointConfig(BaseModel):
    """Model checkpoint configuration"""

    name: str
    path: str
    backbone: C.ModelBackbone


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""

    evalset_dir: str
    mapping_path: str
    models: list[ModelCheckpointConfig]


class ExportConfig(BaseModel):
    """Export configuration"""

    save_last_weight: bool
    save_best_weight: bool
    export_last_weight: bool
    export_best_weight: bool


class PipelineConfig(BaseModel):
    """Pipeline configuration"""

    experiment_label: str
    enable_training: bool
    enable_evaluation: bool
    enable_export: bool
    model: ModelConfig
    dataloader: DataloaderConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    export: ExportConfig