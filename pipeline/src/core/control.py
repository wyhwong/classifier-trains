from glob import glob
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.datasets as datasets

import core.data
import core.model
import core.utils
import core.visualization
import env
import logger
import schemas.constants


torch.manual_seed(env.RANDOM_SEED)
np.random.seed(env.RANDOM_SEED)
local_logger = logger.get_logger(__name__)


class ModelFacade:
    """
    Facade class for the model
    """

    def __init__(self, setting: dict[str, Any]) -> None:
        """
        Constructor for the model facade

        Args:
        -----
            setting (dict[str, Any]):
                Setting for the model

        Returns:
        -----
            None
        """

        local_logger.info("Initializing model facade: %s", setting)

        self._setting = setting
        self._output_dir: str = ""
        self._data_transforms: dict[schemas.constants.Phase, Any] = {}
        self._dataloaders: dict[schemas.constants.Phase, torch.utils.data.DataLoader] = {}
        self._model: torchvision.models = None
        self._best_weights: dict[str, Any] = {}
        self._last_weights: dict[str, Any] = {}
        self._loss = pd.DataFrame()
        self._accuracy = pd.DataFrame()

        self._initialize()
        self._initialize_data_transform()
        self._initialize_dataset()
        self._initialize_model()

    def _initialize(self) -> None:
        """
        Initialize the model facade

        Args:
        -----
            None

        Returns:
        -----
            None
        """

        core.utils.check_and_create_dir(env.RESULT_DIR)
        num_experiment = len(glob(f"{env.RESULT_DIR}/*")) + 1
        self._output_dir = f"{env.RESULT_DIR}/{num_experiment}"
        if self._setting["experiment_label"]:
            self._output_dir += "_" + self._setting["experiment_label"]
        core.utils.check_and_create_dir(self._output_dir)
        core.utils.save_as_yml(f"{self._output_dir}/setting.yml", self._setting)

        local_logger.info("Initialized output directory: %s", self._output_dir)

    def _initialize_model(self) -> None:
        """
        Initialize the model

        Args:
        -----
            None

        Returns:
        -----
            None
        """

        self._model = core.model.initialize.initialize_model(
            backbone=schemas.constants.ModelBackbone(self._setting["model"]["backbone"]),
            weights=self._setting["model"]["weights"],
            num_classes=self._setting["model"]["num_classes"],
            unfreeze_all_params=self._setting["model"]["unfreeze_all_params"],
        )

        local_logger.info("Initialized model.")

    def _initialize_data_transform(self) -> None:
        """
        Initialize the data transform

        Args:
        -----
            None

        Returns:
        -----
            None
        """

        spatial_augmentation = core.data.preprocessing.get_spatial_transforms(
            width=self._setting["preprocessing"]["width"],
            height=self._setting["preprocessing"]["height"],
            hflip_prob=self._setting["preprocessing"]["spatial"]["hflip_prob"],
            vflip_prob=self._setting["preprocessing"]["spatial"]["vflip_prob"],
            max_rotate=self._setting["preprocessing"]["spatial"]["max_rotate"],
            centor_crop=self._setting["preprocessing"]["spatial"]["centor_crop"],
            random_crop=self._setting["preprocessing"]["spatial"]["random_crop"],
        )
        color_augmentation = core.data.preprocessing.get_color_transforms(
            gray_scale=self._setting["preprocessing"]["color"]["gray_scale"],
            random_color_augmentation=self._setting["preprocessing"]["color"]["random_color_augmentation"],
        )
        resize_and_padding = core.data.preprocessing.get_resize_and_padding_transforms(
            width=self._setting["preprocessing"]["width"],
            height=self._setting["preprocessing"]["height"],
            maintain_aspect_ratio=self._setting["preprocessing"]["resize_and_padding"]["maintain_aspect_ratio"],
            interpolation=schemas.constants.InterpolationType(
                self._setting["preprocessing"]["resize_and_padding"]["interpolation"]
            ),
            padding=schemas.constants.PaddingType(self._setting["preprocessing"]["resize_and_padding"]["padding"]),
        )
        self._data_transforms = core.data.preprocessing.get_transforms(
            spatial_augmentation=spatial_augmentation,
            color_augmentation=color_augmentation,
            resize_and_padding=resize_and_padding,
            mean=self._setting["preprocessing"]["mean"],
            std=self._setting["preprocessing"]["std"],
        )

        local_logger.info("Initialized data transforms.")

    def _initialize_dataset(self) -> None:
        """
        Initialize the dataset

        Args:
        -----
            None

        Returns:
        -----
            None
        """

        for phase in [schemas.constants.Phase.TRAINING, schemas.constants.Phase.VALIDATION]:
            image_dataset = datasets.ImageFolder(
                self._setting["dataset"][f"{phase.value}set_dir"],
                self._data_transforms[phase],
            )
            core.visualization.dataset.get_dataset_preview(
                dataset=image_dataset,
                mean=self._setting["preprocessing"]["mean"],
                std=self._setting["preprocessing"]["std"],
                filename=f"preview_{phase.value}.png",
                output_dir=self._output_dir,
            )

            self._dataloaders[phase] = torch.utils.data.DataLoader(
                image_dataset,
                shuffle=True,
                batch_size=self._setting["dataset"]["batch_size"],
                num_workers=self._setting["dataset"]["num_workers"],
            )

        core.data.mapping.save_class_mapping(
            dataset=image_dataset,
            savepath=f"{self._output_dir}/class_mapping.yml",
        )

    def _run_training(self) -> None:
        """
        Run the training phase

        Args:
        -----
            None

        Returns:
        -----
            None
        """

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = core.model.initialize.initialize_optimizer(
            params=self._model.parameters(),
            optimizier=schemas.constants.OptimizerType(self._setting["training"]["optimizer"]["name"]),
            lr=self._setting["training"]["optimizer"]["lr"],
            momentum=self._setting["training"]["optimizer"]["momentum"],
            weight_decay=self._setting["training"]["optimizer"]["weight_decay"],
            alpha=self._setting["training"]["optimizer"]["alpha"],
            betas=self._setting["training"]["optimizer"]["betas"],
        )
        scheduler = core.model.initialize.initialize_scheduler(
            scheduler=schemas.constants.SchedulerType(self._setting["training"]["scheduler"]["name"]),
            optimizer=optimizer,
            num_epochs=self._setting["training"]["num_epochs"],
            step_size=self._setting["training"]["scheduler"]["step_size"],
            gamma=self._setting["training"]["scheduler"]["gamma"],
            lr_min=self._setting["training"]["scheduler"]["lr_min"],
        )
        self._model, self._best_weights, self._last_weights, loss, accuracy = core.model.train.train_model(
            model=self._model,
            dataloaders=self._dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=self._setting["training"]["num_epochs"],
            best_criteria=schemas.constants.BestCriteria(self._setting["training"]["best_criteria"]),
        )
        self._loss = pd.DataFrame(loss)
        core.visualization.performance.loss_curve(df=self._loss, output_dir=self._output_dir)
        self._accuracy = pd.DataFrame(accuracy)
        core.visualization.performance.accuracy_curve(df=self._accuracy, output_dir=self._output_dir)

        local_logger.info("Training phase ended.")

    def _run_evaluation(self) -> None:
        """
        Run the evaluation phase

        Args:
        -----
            None

        Returns:
        -----
            None
        """

        image_dataset = datasets.ImageFolder(
            self._setting["evaluation"]["evalset_dir"],
            self._data_transforms[schemas.constants.Phase.VALIDATION],
        )
        dataloader = torch.utils.data.DataLoader(
            image_dataset,
            shuffle=False,
            batch_size=self._setting["dataset"]["batch_size"],
            num_workers=self._setting["dataset"]["num_workers"],
        )
        core.visualization.dataset.get_dataset_preview(
            dataset=image_dataset,
            mean=self._setting["preprocessing"]["mean"],
            std=self._setting["preprocessing"]["std"],
            filename="preview_eval.png",
            output_dir=self._output_dir,
        )

        mapping_path = self._setting["evaluation"]["mapping_path"].replace("$OUTPUT_DIR", self._output_dir)
        mapping = core.utils.load_yml(mapping_path)
        models = []
        model_names = []
        for model_for_eval in self._setting["evaluation"]["models"]:
            weights_path = model_for_eval["path"].replace("$OUTPUT_DIR", self._output_dir)
            model = core.model.initialize.initialize_model(
                backbone=schemas.constants.ModelBackbone(model_for_eval["backbone"]),
                weights="DEFAULT",
                num_classes=len(mapping.keys()),
                unfreeze_all_params=False,
            )
            core.model.load.load_model(model, weights_path)
            models.append(model)
            model_names.append(model_for_eval["name"])

            y_true, y_pred, _ = core.model.inference.predict(model, dataloader)
            core.visualization.performance.confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                output_dir=self._output_dir,
                filename=f"confusion_matrix_{model_for_eval['name']}.png",
            )

        core.visualization.performance.roc_curves(
            models=models,
            model_names=model_names,
            dataloader=dataloader,
            mapping=mapping,
            output_dir=self._output_dir,
        )

        local_logger.info("Evaluation phase ended.")

    def _run_export(self) -> None:
        """
        Run the export phase

        Args:
        -----
            None

        Returns:
        -----
            None
        """

        if self._setting["export"]["save_last_weight"]:
            model_path = f"{self._output_dir}/last_model.pt"
            core.model.load.save_weights(self._last_weights, model_path)

        if self._setting["export"]["save_best_weight"]:
            model_path = f"{self._output_dir}/best_model.pt"
            core.model.load.save_weights(self._best_weights, model_path)

        if self._setting["export"]["export_last_weight"]:
            model_path = f"{self._output_dir}/last_model.onnx"
            self._model.load_state_dict(self._last_weights)
            core.model.load.export_model_to_onnx(
                model=self._model,
                input_height=self._setting["preprocessing"]["height"],
                input_width=self._setting["preprocessing"]["width"],
                export_path=model_path,
            )
            core.model.load.check_model_is_valid(model_path)

        if self._setting["export"]["export_best_weight"]:
            model_path = f"{self._output_dir}/best_model.onnx"
            self._model.load_state_dict(self._best_weights)
            core.model.load.export_model_to_onnx(
                model=self._model,
                input_height=self._setting["preprocessing"]["height"],
                input_width=self._setting["preprocessing"]["width"],
                export_path=model_path,
            )
            core.model.load.check_model_is_valid(model_path)

        local_logger.info("Export phase ended.")

    def start(self) -> None:
        """
        Start the model facade

        Args:
        -----
            None

        Returns:
        --------
            None
        """

        if self._setting["setup"]["enable_training"]:
            self._run_training()

        if self._setting["setup"]["enable_export"]:
            self._run_export()

        if self._setting["setup"]["enable_evaluation"]:
            self._run_evaluation()

        local_logger.info("Model facade ended.")
