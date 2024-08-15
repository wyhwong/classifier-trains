import datetime

import onnx
import torch

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
            self.__model.load_state_dict(torch.load(self.__model_config.weights_path))

    def __load_weights(self, weights_path: str) -> None:
        """Load the weights from the given path.

        Args:
            weights_path (str): The path to the weights file.
        """

        weights = torch.load(weights_path)
        self.__model.load_state_dict(weights)
        local_logger.info("Loaded weights from local file: %s.", weights_path)

    def __export_weights(self, weights_path: str) -> None:
        """Export the model weights.

        Args:
            weights_path (str): The path to export the weights.
        """

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
        device="cuda",
    ) -> None:
        """Export the model to ONNX format.

        Args:
            onnx_path (str): The path to export the ONNX model
            dim (tuple[int, int]): The dimensions of the input
            device (str, optional): The device to export the model. Defaults to "cuda".
        """

        self.__model.eval()
        _height, _width = dim
        x = torch.randn(1, 3, _height, _width, requires_grad=True)
        torch.onnx.export(
            self.__model,
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

    def inference(self, data: torch.Tensor) -> torch.Tensor:
        """Perform inference

        Args:
            data (torch.Tensor): The input data

        Returns:
            torch.Tensor: The output of the model
        """

        self.__model.eval()
        confidence = self.__model(data)
        return torch.max(torch.exp(confidence), 1)[1]
