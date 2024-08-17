import datetime
from copy import deepcopy
from typing import Optional

import numpy as np
import onnx
import torch
from torch import nn
from tqdm import tqdm

import pipeline.core.model.utils
import pipeline.logger
from pipeline.schemas import config, constants


local_logger = pipeline.logger.get_logger(__name__)


class ModelFacade:
    """Class to handle the model related functions.

    TODO: Migrate to pytorch lightning
    """

    def __init__(self, model_config: config.ModelConfig) -> None:
        """Initialize the ModelFacade object.

        Args:
            model_config (config.ModelConfig): The model configuration
        """

        local_logger.info("Initializing ModelFacade with config %s", model_config)

        self.__model_config = model_config
        self.__model = pipeline.core.model.utils.initialize_model(
            model_config=self.__model_config,
        )

        if self.__model_config.weights_path:
            self.__load_weights(self.__model_config.weights_path)

        self.__best_weights = deepcopy(self.__model.state_dict())
        self.__training_history = {
            criterion: {
                constants.Phase.TRAINING: [],
                constants.Phase.VALIDATION: [],
            }
            for criterion in constants.BestCriteria
        }

    def __load_weights(self, weights_path: str) -> None:
        """Load the weights from the given path.

        Args:
            weights_path (str): The path to the weights file.
        """

        weights = torch.load(weights_path)
        self.__model.load_state_dict(weights)
        local_logger.info("Loaded weights from local file: %s.", weights_path)

    def __export_weights(self, weights_path: str, is_best=False) -> None:
        """Export the model weights.

        Args:
            weights_path (str): The path to export the weights.
            is_best (bool, optional): Export the best weights. Defaults to False.
        """

        if is_best:
            torch.save(self.__best_weights, weights_path)
        else:
            torch.save(self.__model.state_dict(), weights_path)

        local_logger.info("Saved model weights at %s.", weights_path)

    def __check_exported_model(self, onnx_path: str) -> None:
        """Check if the model is exported successfully.

        Args:
            onnx_path (str): The path to the exported model.
        """

        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        local_logger.info("Model is exported successfully.")

    def export_onnx(
        self,
        onnx_path: str,
        dim: tuple[int, int],
        is_best=False,
        device="cuda",
    ) -> None:
        """Export the model to ONNX format.

        Args:
            onnx_path (str): The path to export the ONNX model
            dim (tuple[int, int]): The dimensions of the input
            is_best (bool, optional): Export the best model. Defaults to False.
            device (str, optional): The device to export the model. Defaults to "cuda".
        """

        model = deepcopy(self.__model)

        if is_best:
            model.load_state_dict(self.__best_weights)

        model.eval()
        _height, _width = dim
        x = torch.randn(1, 3, _height, _width, requires_grad=True)
        torch.onnx.export(
            model,
            x.to(device),
            onnx_path,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )

        local_logger.info("Exported the model at %s.", onnx_path)

    def train(
        self,
        dataloaders: dict[constants.Phase, torch.utils.data.DataLoader],
        num_epochs: int,
        best_criteria: constants.BestCriteria,
        optimizer_config: config.OptimizerConfig,
        scheduler_config: config.SchedulerConfig,
    ) -> None:
        """Train the model.

        Args:
            num_epochs (int): The number of epochs to train the model
            best_criteria (constants.BestCriteria): The criteria to select the best model
            optimizer_config (config.OptimizerConfig): The optimizer configuration
            scheduler_config (config.SchedulerConfig): The scheduler configuration
            export_last_weight (bool): Export the last weight
            export_last_as_onnx (bool): Export the last weight as ONNX
            export_best_weight (bool): Export the best weight
            export_best_as_onnx (bool): Export the best weight as ONNX
        """

        training_start = datetime.datetime.now(tz=datetime.timezone.utc)
        local_logger.info("Training started at %s.", training_start)

        optimizer = pipeline.core.model.utils.initialize_optimizer(
            params=self.__model.parameters(),
            optimizer_config=optimizer_config,
        )
        scheduler = pipeline.core.model.utils.initialize_scheduler(
            optimizer=optimizer,
            scheduler_config=scheduler_config,
            num_epochs=num_epochs,
        )
        loss_fn = nn.CrossEntropyLoss()
        best_record = np.inf if best_criteria is constants.BestCriteria.LOSS else -np.inf

        # Start training
        for epoch in range(1, num_epochs + 1):
            local_logger.info("-" * 40)
            local_logger.info("Epoch %d/%d", epoch, num_epochs)
            local_logger.info("-" * 20)

            for phase in [constants.Phase.TRAINING, constants.Phase.VALIDATION]:
                if phase is constants.Phase.TRAINING:
                    local_logger.debug("The %d-th epoch training started.", epoch)
                    self.__model.train()
                else:
                    local_logger.debug("The %d-th epoch validation started.", epoch)
                    self.__model.eval()

                epoch_loss = 0.0
                epoch_corrects = 0
                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(pipeline.env.DEVICE)
                    labels = labels.to(pipeline.env.DEVICE)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase is constants.Phase.TRAINING):
                        outputs = self.__model(inputs.float())
                        prediction_loss = loss_fn(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase is constants.Phase.TRAINING:
                            prediction_loss.backward()
                            optimizer.step()

                    epoch_loss += prediction_loss.item() * inputs.size(0)
                    epoch_corrects += int(torch.sum(preds == labels.data))

                epoch_loss = epoch_loss / len(dataloaders[phase].dataset)
                epoch_acc = epoch_corrects / len(dataloaders[phase].dataset)

                if scheduler and phase is constants.Phase.TRAINING:
                    scheduler.step()
                    local_logger.info("Last learning rate in this epoch: %.3f", scheduler.get_last_lr()[0])

                local_logger.info("%s Loss: %.4f Acc: %.4f.", phase, epoch_loss, epoch_acc)

                if phase is constants.Phase.VALIDATION:
                    if best_criteria is constants.BestCriteria.LOSS and epoch_loss < best_record:
                        local_logger.info("New Record: %.4f < %.4f", epoch_loss, best_record)
                        best_record = epoch_loss
                        self.__best_weights = deepcopy(self.__model.state_dict())
                        local_logger.debug("Updated best models.")

                    if best_criteria is constants.BestCriteria.ACCURACY and epoch_acc > best_record:
                        local_logger.info("New Record: %.4f < %.4f", epoch_acc, best_record)
                        best_record = epoch_acc
                        self.__best_weights = deepcopy(self.__model.state_dict())
                        local_logger.debug("Updated best models.")

                self.__training_history[constants.BestCriteria.LOSS][phase].append(float(epoch_loss))
                self.__training_history[constants.BestCriteria.ACCURACY][phase].append(float(epoch_acc))

                local_logger.debug(
                    "Updated %s accuracy: %.4f, loss: %.4f",
                    phase,
                    epoch_acc,
                    epoch_loss,
                )

        time_elapsed = datetime.datetime.now(tz=datetime.timezone.utc) - training_start
        local_logger.info("Training complete at %s", datetime.now())
        local_logger.info("Training complete in %dm %ds.", time_elapsed // 60, time_elapsed % 60)
        local_logger.info("Best val %s: %.4f}.", best_criteria, best_record)

    def inference(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform inference

        Args:
            data (torch.Tensor): The input data

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The predicted class and confidence
        """

        self.__model.eval()
        confidence = self.__model(data)
        return (torch.max(torch.exp(confidence), 1)[1], confidence)

    def export(
        self,
        output_dir: str,
        export_last_weight: bool,
        export_best_weight: bool,
        export_last_as_onnx: bool,
        export_best_as_onnx: bool,
        dim: Optional[tuple[int, int]],
        device: str = "cuda",
    ):
        """Export the trained model"""

        if export_best_weight:
            self.__export_weights(f"{output_dir}/best.pt", is_best=True)

        if export_best_as_onnx:
            onnx_path = f"{output_dir}/best.onnx"
            self.export_onnx(onnx_path, dim, is_best=True, device=device)
            self.__check_exported_model(onnx_path)

        if export_last_weight:
            self.__export_weights(f"{output_dir}/last.pt", is_best=False)

        if export_last_as_onnx:
            onnx_path = f"{output_dir}/last.onnx"
            self.export_onnx(onnx_path, dim, is_best=False, device=device)
            self.__check_exported_model(onnx_path)

    def get_training_history(self):
        """Get the training history"""

        return self.__training_history
