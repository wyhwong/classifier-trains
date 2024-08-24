from typing import Literal, Optional

from pydantic import BaseModel, NonNegativeFloat, PositiveInt, field_validator

import pipeline.schemas.constants as C


class ModelConfig(BaseModel):
    """Model configuration"""

    backbone: C.ModelBackbone
    num_classes: PositiveInt
    weights: str
    checkpoint_path: Optional[str] = None


class DataloaderConfig(BaseModel):
    """Dataset configuration"""

    trainset_dir: str
    valset_dir: str
    testset_dir: Optional[str]
    batch_size: PositiveInt
    num_workers: PositiveInt

    def get_dirpath(self, phase: C.Phase) -> str:
        """Get the directory path for the given phase.

        Args:
            phase (C.Phase): The phase.

        Returns:
            str: The directory path.
        """

        if phase is C.Phase.TRAINING:
            return self.trainset_dir

        return self.valset_dir


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
    allow_center_crop: bool
    allow_random_crop: bool

    @field_validator("hflip_prob")
    @classmethod
    def hfilp_prob_is_valid(cls, v: float) -> float:
        """Validate the hflip probability."""

        if v < 0 or v > 1:
            raise ValueError("hflip_prob must be between 0 and 1")
        return v

    @field_validator("vflip_prob")
    @classmethod
    def vfilp_prob_is_valid(cls, v: float) -> float:
        """Validate the vflip probability."""

        if v < 0 or v > 1:
            raise ValueError("vflip_prob must be between 0 and 1")
        return v

    @field_validator("max_rotate_in_degree")
    @classmethod
    def max_rotate_in_degree_is_valid(cls, v: float) -> float:
        """Validate the maximum rotation in degree."""

        if v < 0 or v > 180:
            raise ValueError("max_rotate_in_degree must be between 0 and 180")
        return v


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

    name: str
    num_epochs: PositiveInt
    random_seed: PositiveInt
    precision: Literal[64, 32, 16]
    device: str
    max_num_hrs: Optional[NonNegativeFloat]
    validate_every_n_epoch: PositiveInt
    save_every_n_epoch: PositiveInt
    patience_in_epoch: PositiveInt
    criterion: C.Criterion
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    export_last_as_onnx: bool
    export_best_as_onnx: bool


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""

    name: str
    device: str
    random_seed: PositiveInt
    precision: Literal[64, 32, 16]
    evalset_dir: str
    models: list[ModelConfig]
