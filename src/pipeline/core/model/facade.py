import datetime
from typing import Optional

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

import pipeline.core.model.classifier
import pipeline.core.model.utils
import pipeline.logger
from pipeline.schemas import config, constants


local_logger = pipeline.logger.get_logger(__name__)


class ModelFacade:
    """Class to handle the model related functions."""

    def __init__(self, model_config: config.ModelConfig) -> None:
        """Initialize the ModelFacade object.

        Args:
            model_config (config.ModelConfig): The model configuration
        """

        local_logger.info("Initializing ModelFacade with config %s", model_config)

        self.__model = pipeline.core.model.classifier.ClassifierModel(model_config=model_config)
        self.__trainer: Optional[pl.Trainer] = None

    def train(
        self,
        training_config: config.TrainingConfig,
        datamodule: pl.LightningDataModule,
        output_dir: str,
    ) -> None:
        """Train the model.

        Args:
            training_config (config.TrainingConfig): The training configuration
            datamodule (pl.LightningDataModule): The datamodule
            output_dir (str): The output directory
        """

        pl.pytorch.seed_everything(training_config.random_seed)

        self.__model.training_setup(
            num_epochs=training_config.num_epochs,
            optimizer_config=training_config.optimizer,
            scheduler_config=training_config.scheduler,
        )
        version = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root_path = f"{output_dir}/{training_config.name}/{version}"
        mode = "min" if training_config.criterion is constants.Criterion.LOSS else "max"
        max_time = datetime.timedelta(hours=training_config.max_num_hrs) if training_config.max_num_hrs else None

        self.__trainer = pl.pytorch.Trainer(
            precision=training_config.precision,
            accelerator=training_config.device,
            max_epochs=training_config.num_epochs,
            max_time=max_time,
            check_val_every_n_epoch=training_config.validate_every_n_epoch,
            log_every_n_steps=1,
            default_root_dir=root_path,
            callbacks=[
                LearningRateMonitor(
                    logging_interval="step",
                    log_momentum=True,
                    log_weight_decay=True,
                ),
                EarlyStopping(
                    monitor=constants.Phase.VALIDATION(training_config.criterion),
                    patience=training_config.patience_in_epoch,
                    verbose=True,
                    mode=mode,
                ),
                ModelCheckpoint(
                    monitor=constants.Phase.VALIDATION(training_config.criterion),
                    dirpath=root_path,
                    filename=training_config.name + "-best-{epoch:02d}",
                    save_top_k=1,
                    save_last=True,
                    verbose=True,
                    mode=mode,
                ),
                ModelCheckpoint(
                    dirpath=root_path,
                    filename=training_config.name + "-{epoch:02d}",
                    every_n_epochs=training_config.save_every_n_epoch,
                    save_top_k=-1,
                    verbose=True,
                ),
            ],
        )

        self.__trainer.fit(
            model=self.__model,
            datamodule=datamodule,
        )

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Perform inference

        Args:
            x (torch.Tensor): The input data

        Returns:
            torch.Tensor: The output data
        """

        return self.__model.forward(x)
