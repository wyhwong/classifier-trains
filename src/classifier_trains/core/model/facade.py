import datetime
from typing import Callable, Optional

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from classifier_trains.core.model.classifier import ClassifierModel
from classifier_trains.schemas import config, constants
from classifier_trains.utils import file, logger


local_logger = logger.get_logger(__name__)


class ModelFacade:
    """Class to handle the model related functions."""

    def __init__(self, denorm_fn: Optional[Callable] = None) -> None:
        """Initialize the ModelFacade object.

        Args:
            denorm_fn (Optional[Callable], optional): The denormalization function. Defaults to None.
        """

        self._denorm_fn = denorm_fn

        local_logger.info("Initializing ModelFacade.")

    def train(
        self,
        model_config: config.ModelConfig,
        training_config: config.TrainingConfig,
        datamodule: pl.LightningDataModule,
        output_dir: str,
        input_sample: Optional[torch.Tensor] = None,
    ) -> None:
        """Train the model.

        Args:
            training_config (config.TrainingConfig): The training configuration
            datamodule (pl.LightningDataModule): The datamodule
            output_dir (str): The output directory
            input_sample (Optional[torch.Tensor], optional): The input sample for onnx export,
                Defaults to None.
        """

        local_logger.info("Training with config: %s", training_config)
        local_logger.info("Model config: %s", model_config)

        if not model_config.checkpoint_path:
            model = ClassifierModel(model_config=model_config, denorm_fn=self._denorm_fn)
        else:
            # NOTE: Here we need to disable the pylint check for no-value-for-parameter
            # Expected message: E1120: No value for argument 'cls' in unbound method call
            model = ClassifierModel.load_from_checkpoint(  # pylint: disable=E1120
                checkpoint_path=model_config.checkpoint_path,
                model_config=model_config,
                denorm_fn=self._denorm_fn,
            )

        pl.pytorch.seed_everything(training_config.random_seed)

        model.training_setup(
            num_epochs=training_config.num_epochs,
            optimizer_config=training_config.optimizer,
            scheduler_config=training_config.scheduler,
            input_sample=input_sample,
        )
        mode = "min" if training_config.criterion is constants.Criterion.LOSS else "max"
        max_time = datetime.timedelta(hours=training_config.max_num_hrs) if training_config.max_num_hrs else None
        name = f"train-{training_config.name}"
        version = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root_dir = f"{output_dir}/{name}/{version}"

        local_logger.info("Training results will be logged at %s", root_dir)

        trainer_logger = TensorBoardLogger(save_dir=output_dir, version=version, name=name, log_graph=True)
        trainer = pl.pytorch.Trainer(
            logger=trainer_logger,
            precision=training_config.precision,
            accelerator=training_config.device,
            max_epochs=training_config.num_epochs,
            max_time=max_time,
            check_val_every_n_epoch=training_config.validate_every,
            log_every_n_steps=1,
            default_root_dir=root_dir,
            callbacks=[
                LearningRateMonitor(
                    logging_interval="step",
                    log_momentum=True,
                    log_weight_decay=True,
                ),
                EarlyStopping(
                    monitor=constants.Phase.VALIDATION(training_config.criterion),
                    patience=training_config.patience,
                    verbose=True,
                    mode=mode,
                ),
                ModelCheckpoint(
                    monitor=constants.Phase.VALIDATION(training_config.criterion),
                    dirpath=root_dir,
                    filename="best",
                    save_top_k=1,
                    save_last=True,
                    verbose=True,
                    mode=mode,
                ),
                ModelCheckpoint(
                    dirpath=root_dir,
                    filename=training_config.name + "-{epoch:02d}",
                    every_n_epochs=training_config.save_every,
                    save_top_k=-1,
                    verbose=True,
                ),
            ],
        )

        trainer.fit(model=model, datamodule=datamodule)
        file.save_as_yml(f"{root_dir}/training.yml", training_config.model_dump())

        if training_config.export_best_as_onnx:
            self._export_checkpoint_as_onnx(model_config, "best.ckpt", root_dir, input_sample)

        if training_config.export_last_as_onnx:
            self._export_checkpoint_as_onnx(model_config, "last.ckpt", root_dir, input_sample)

    def _export_checkpoint_as_onnx(
        self,
        model_config: config.ModelConfig,
        checkpoint_name: str,
        root_dir: str,
        input_sample: Optional[torch.Tensor] = None,
    ) -> None:
        """Export the checkpoint as ONNX.

        Args:
            checkpoint_name (str): The checkpoint name
            root_dir (str): The root directory
            input_sample (Optional[torch.Tensor], optional): The input sample for onnx export,
                Defaults to None.
        """

        local_logger.info("Exporting checkpoint %s as ONNX", checkpoint_name)

        model = ClassifierModel.load_from_checkpoint(  # pylint: disable=E1120
            checkpoint_path=f"{root_dir}/{checkpoint_name}",
            model_config=model_config,
        )
        model.to_onnx(
            f"{root_dir}/{checkpoint_name.replace('.ckpt', '.onnx')}",
            input_sample=input_sample,
        )

    @staticmethod
    def evaluate(
        evaluation_config: config.EvaluationConfig,
        dataloader: torch.utils.data.DataLoader,
        output_dir: str,
    ) -> None:
        """Evaluate trained models.

        Args:
            evaluation_config (config.EvaluationConfig): The evaluation configuration
            dataloader (torch.utils.data.DataLoader): The dataloader
            output_dir (str): The output directory
        """

        local_logger.info("Evaluating with config %s", evaluation_config)

        pl.pytorch.seed_everything(evaluation_config.random_seed)

        name = f"eval-{evaluation_config.name}"
        version = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root_dir = f"{output_dir}/{name}/{version}"

        local_logger.info("Evaluation results will be logged at %s", root_dir)

        trainer_logger = TensorBoardLogger(save_dir=output_dir, version=version, name=name)
        trainer = pl.pytorch.Trainer(
            logger=trainer_logger,
            precision=evaluation_config.precision,
            accelerator=evaluation_config.device,
            default_root_dir=root_dir,
        )
        for model_config in evaluation_config.models:
            if model_config.checkpoint_path:
                model = ClassifierModel.load_from_checkpoint(  # pylint: disable=E1120
                    checkpoint_path=model_config.checkpoint_path,
                    model_config=model_config,
                )
            else:
                model = ClassifierModel(model_config=model_config)

            trainer.test(model=model, dataloaders=dataloader)

        file.save_as_yml(f"{root_dir}/evaluation.yml", evaluation_config.model_dump())
