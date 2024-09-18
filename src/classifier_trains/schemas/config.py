import os
from typing import Literal, Optional

from pydantic import BaseModel, NonNegativeFloat, PositiveFloat, PositiveInt, field_validator, model_validator

import classifier_trains.schemas.constants as C


class ModelConfig(BaseModel):
    """Model configuration"""

    backbone: C.ModelBackbone
    num_classes: PositiveInt
    weights: str
    checkpoint_path: Optional[str] = None

    @field_validator("checkpoint_path")
    @classmethod
    def checkpoint_path_is_valid(cls, v: str) -> str:
        """Validate the checkpoint path."""

        if v is not None and not v.endswith(".ckpt"):
            raise ValueError("checkpoint_path must be a .ckpt file")

        if v is not None and not os.path.isfile(v):
            raise ValueError("checkpoint_path must be a valid file")

        return v


class DataloaderConfig(BaseModel):
    """Dataset configuration"""

    trainset_dir: str
    valset_dir: str
    batch_size: PositiveInt
    num_workers: PositiveInt
    testset_dir: Optional[str] = None

    @model_validator(mode="after")
    @classmethod
    def trainset_dir_is_valid(cls, v):
        """Validate the training set directory."""

        if not os.path.isdir(v.trainset_dir):
            raise ValueError(f"trainset_dir {v.trainset_dir} does not exist.")

        return v

    @model_validator(mode="after")
    @classmethod
    def valset_dir_is_valid(cls, v):
        """Validate the validation set directory."""

        if not os.path.isdir(v.valset_dir):
            raise ValueError(f"valset_dir {v.valset_dir} does not exist.")

        return v

    @model_validator(mode="after")
    @classmethod
    def testset_dir_is_valid(cls, v: Optional[str]) -> Optional[str]:
        """Validate the test set directory."""

        if v.testset_dir and not os.path.isdir(v.testset_dir):
            raise ValueError(f"testset_dir {v.testset_dir} does not exist.")

        return v

    def get_dirpath(self, phase: C.Phase) -> Optional[str]:
        """Get the directory path for the given phase.

        Args:
            phase (C.Phase): The phase.

        Returns:
            str: The directory path.
        """

        if phase is C.Phase.TRAINING:
            return self.trainset_dir

        if phase is C.Phase.VALIDATION:
            return self.valset_dir

        return self.testset_dir


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
    lr: PositiveFloat
    weight_decay: NonNegativeFloat
    momentum: Optional[NonNegativeFloat] = None
    alpha: Optional[NonNegativeFloat] = None
    betas: Optional[tuple[NonNegativeFloat, NonNegativeFloat]] = None

    @model_validator(mode="after")
    @classmethod
    def validate_adam(cls, v):
        """Validate the Adam optimizer configuration."""

        if v.name is C.OptimizerType.ADAM:
            if v.momentum:
                raise ValueError("momentum is expected to be None for Adam optimizer")

            if v.alpha:
                raise ValueError("alpha is expected to be None for Adam optimizer")

        return v

    @model_validator(mode="after")
    @classmethod
    def validate_adamw(cls, v):
        """Validate the AdamW optimizer configuration."""

        if v.name is C.OptimizerType.ADAMW:
            if v.momentum:
                raise ValueError("momentum is expected to be None for AdamW optimizer")

            if v.alpha:
                raise ValueError("alpha is expected to be None for AdamW optimizer")

        return v

    @model_validator(mode="after")
    @classmethod
    def validate_sgd(cls, v):
        """Validate the SGD optimizer configuration."""

        if v.name is C.OptimizerType.SGD:
            if v.alpha:
                raise ValueError("alpha is expected to be None for SGD optimizer")

            if v.betas:
                raise ValueError("betas is expected to be None for SGD optimizer")

        return v

    @model_validator(mode="after")
    @classmethod
    def validate_rmsprop(cls, v):
        """Validate the RMSprop optimizer configuration."""

        if v.name is C.OptimizerType.RMSPROP:
            if v.betas:
                raise ValueError("betas is expected to be None for RMSprop optimizer")

        return v


class SchedulerConfig(BaseModel):
    """Scheduler configuration"""

    name: C.SchedulerType
    lr_min: Optional[NonNegativeFloat] = None
    step_size: Optional[PositiveInt] = None
    gamma: Optional[NonNegativeFloat] = None

    @model_validator(mode="after")
    @classmethod
    def validate_step(cls, v):
        """Validate the StepLR scheduler configuration."""

        if v.name is C.SchedulerType.STEP:
            if v.lr_min:
                raise ValueError("lr_min is expected to be None for StepLR scheduler")

            if v.gamma is None:
                raise ValueError("gamma is expected to be input for StepLR scheduler")

            if v.step_size is None:
                raise ValueError("step_size is expected to be input for StepLR scheduler")

        return v

    @model_validator(mode="after")
    @classmethod
    def validate_cosine(cls, v):
        """Validate the CosineAnnealingLR scheduler configuration."""

        if v.name is C.SchedulerType.COSINE:
            if v.lr_min is None:
                raise ValueError("lr_min is expected to be input for CosineAnnealingLR scheduler")

            if v.gamma:
                raise ValueError("gamma is expected to be None for CosineAnnealingLR scheduler")

            if v.step_size:
                raise ValueError("step_size is expected to be None for CosineAnnealingLR scheduler")

        return v


class TrainingConfig(BaseModel):
    """Training configuration"""

    name: str
    num_epochs: PositiveInt
    random_seed: PositiveInt
    precision: Literal[64, 32, 16]
    device: str
    validate_every_n_epoch: PositiveInt
    save_every_n_epoch: PositiveInt
    patience_in_epoch: PositiveInt
    criterion: C.Criterion
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    export_last_as_onnx: bool
    export_best_as_onnx: bool
    max_num_hrs: Optional[NonNegativeFloat] = None


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""

    name: str
    device: str
    random_seed: PositiveInt
    precision: Literal[64, 32, 16]
    evalset_dir: str
    models: list[ModelConfig]
