import datetime
from typing import Callable, Optional

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pipeline.core.model.classifier import ClassifierModel
from pipeline.schemas import config, constants
from pipeline.utils import file, logger


local_logger = logger.get_logger(__name__)


class ModelFacade:
    """Class to handle the model related functions."""

    def __init__(self, model_config: config.ModelConfig, denorm_fn: Optional[Callable] = None) -> None:
        """Initialize the ModelFacade object.

        Args:
            model_config (config.ModelConfig): The model configuration
            denorm_fn (Optional[Callable], optional): The denormalization function. Defaults to None.
        """

        local_logger.info("Initializing ModelFacade with config %s", model_config)

        self.__model_config = model_config
        self.__version = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        if not model_config.checkpoint_path:
            self.__model = ClassifierModel(
                model_config=model_config,
                denorm_fn=denorm_fn,
            )
        else:
            # NOTE: Here we need to disable the pylint check for no-value-for-parameter
            # Expected message: E1120: No value for argument 'cls' in unbound method call
            self.__model = ClassifierModel.load_from_checkpoint(  # pylint: disable=E1120
                checkpoint_path=model_config.checkpoint_path,
                model_config=model_config,
                denorm_fn=denorm_fn,
            )

    def train(
        self,
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

        pl.pytorch.seed_everything(training_config.random_seed)

        self.__model.training_setup(
            num_epochs=training_config.num_epochs,
            optimizer_config=training_config.optimizer,
            scheduler_config=training_config.scheduler,
            input_sample=input_sample,
        )
        mode = "min" if training_config.criterion is constants.Criterion.LOSS else "max"
        max_time = datetime.timedelta(hours=training_config.max_num_hrs) if training_config.max_num_hrs else None
        name = f"train-{training_config.name}"
        root_dir = f"{output_dir}/{name}/{self.__version}"

        local_logger.info("Training results will be logged at %s", root_dir)

        logger = TensorBoardLogger(save_dir=output_dir, version=self.__version, name=name, log_graph=True)
        trainer = pl.pytorch.Trainer(
            logger=logger,
            precision=training_config.precision,
            accelerator=training_config.device,
            max_epochs=training_config.num_epochs,
            max_time=max_time,
            check_val_every_n_epoch=training_config.validate_every_n_epoch,
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
                    patience=training_config.patience_in_epoch,
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
                    every_n_epochs=training_config.save_every_n_epoch,
                    save_top_k=-1,
                    verbose=True,
                ),
            ],
        )

        trainer.fit(
            model=self.__model,
            datamodule=datamodule,
        )
        file.save_as_yml(f"{root_dir}/training.yml", training_config.model_dump())

        if training_config.export_best_as_onnx:
            self.__export_checkpoint_as_onnx("best.ckpt", root_dir, input_sample)

        if training_config.export_last_as_onnx:
            self.__export_checkpoint_as_onnx("last.ckpt", root_dir, input_sample)

    def __export_checkpoint_as_onnx(
        self,
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

        model = ClassifierModel.load_from_checkpoint(  # pylint: disable=E1120
            checkpoint_path=f"{root_dir}/{checkpoint_name}",
            model_config=self.__model_config,
        )
        model.to_onnx(
            f"{root_dir}/{checkpoint_name.replace('.ckpt', '.onnx')}",
            input_sample=input_sample,
        )

    def evaluate(
        self,
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

        pl.pytorch.seed_everything(evaluation_config.random_seed)

        name = f"eval-{evaluation_config.name}"
        root_dir = f"{output_dir}/{name}/{self.__version}"

        local_logger.info("Evaluation results will be logged at %s", root_dir)

        logger = TensorBoardLogger(save_dir=output_dir, version=self.__version, name=name)
        trainer = pl.pytorch.Trainer(
            logger=logger,
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

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Perform inference

        Args:
            x (torch.Tensor): The input data

        Returns:
            torch.Tensor: The output data
        """

        return self.__model.forward(x)
