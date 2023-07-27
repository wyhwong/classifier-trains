#!/usr/bin/env python3
import torch
import numpy as np
from glob import glob
from torchvision import datasets

from utils.common import get_config, save_dict_as_yml, check_and_create_dir, load_yml
from utils.logger import get_logger, OUTPUT_DIR
from utils.model import initialize_model, load_model, save_weights
from utils.preprocessing import get_transforms
from utils.training import train_model, get_optimizer, get_scheduler, get_class_mapping
from utils.export import export_model_to_onnx, check_model_is_valid
from utils.evaluation import evaluate_model
from utils.visualization import visualize_acc_and_loss, get_dataset_preview

SETUP = get_config()["setup"]
SEED = SETUP["seed"]
TRAIN = SETUP["enable_training"]
EVAL = SETUP["enable_evaluation"]
EXPORT = SETUP["enable_export"]
LOGGER = get_logger("main")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
check_and_create_dir(OUTPUT_DIR)
torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    configs = get_config()
    save_dict_as_yml(f"{OUTPUT_DIR}/train.yml", configs)
    LOGGER.info(f"Initializing training using {DEVICE=}")
    if TRAIN or EVAL:
        data_tranforms = get_transforms(**configs["preprocessing"])

    if TRAIN:
        LOGGER.info("Starting phase: Training, loading necessary parameters.")
        image_datasets = {
            x: datasets.ImageFolder(configs["dataset"][f"{x}set_dir"], data_tranforms[x]) for x in ["train", "val"]
        }
        dataloaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x],
                batch_size=configs["dataset"]["batch_size"],
                shuffle=True,
                num_workers=configs["dataset"]["num_workers"],
            )
            for x in ["train", "val"]
        }
        get_class_mapping(image_datasets["train"], f"{OUTPUT_DIR}/classMapping.yml")
        for phase in ["train", "val"]:
            get_dataset_preview(
                dataset=image_datasets[phase],
                mean=np.array(configs["preprocessing"]["mean"]),
                std=np.array(configs["preprocessing"]["std"]),
                filename_remark=phase,
                output_dir=OUTPUT_DIR,
            )
        criterion = torch.nn.CrossEntropyLoss()
        model = initialize_model(**configs["model"])
        optimizer = get_optimizer(params=model.parameters(), **configs["training"]["optimizer"])
        scheduler = get_scheduler(
            optimizer=optimizer,
            num_epochs=configs["training"]["train_model"]["num_epochs"],
            **configs["training"]["scheduler"],
        )
        LOGGER.info("Loaded all parameters, training starts.")
        model, best_weights, last_weights, train_loss, train_acc = train_model(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **configs["training"]["train_model"],
        )
        LOGGER.info("Training ended, visualizing results.")
        visualize_acc_and_loss(train_loss=train_loss, train_acc=train_acc, output_dir=OUTPUT_DIR)
        LOGGER.info("Training phase ended.")

    if EXPORT:
        LOGGER.info("Starting phase: Export.")
        if configs["export"]["save_last_weight"]:
            save_weights(last_weights, f"{OUTPUT_DIR}/lastModel.pt")
        if configs["export"]["save_best_weight"]:
            save_weights(best_weights, f"{OUTPUT_DIR}/bestModel.pt")
        if configs["export"]["export_last_weight"]:
            model.load_state_dict(last_weights)
            export_model_to_onnx(
                model=model,
                input_height=configs["preprocessing"]["height"],
                input_width=configs["preprocessing"]["width"],
                export_path=f"{OUTPUT_DIR}/lastModel.onnx",
            )
            check_model_is_valid(model_path=f"{OUTPUT_DIR}/lastModel.onnx")
        if configs["export"]["export_best_weight"]:
            model.load_state_dict(best_weights)
            export_model_to_onnx(
                model=model,
                input_height=configs["preprocessing"]["height"],
                input_width=configs["preprocessing"]["width"],
                export_path=f"{OUTPUT_DIR}/bestModel.onnx",
            )
            check_model_is_valid(f"{OUTPUT_DIR}/bestModel.onnx")
        LOGGER.info("Export phase ended.")

    if EVAL:
        LOGGER.info("Starting phase: Evaluation.")
        image_dataset = datasets.ImageFolder(configs["evaluation"][f"evalset_dir"], data_tranforms["val"])
        dataloader = torch.utils.data.DataLoader(
            image_dataset,
            batch_size=configs["dataset"]["batch_size"],
            shuffle=False,
            num_workers=configs["dataset"]["num_workers"],
        )
        model = initialize_model(**configs["model"])
        weights_path = configs["evaluation"]["weights_path"].replace("OUTPUT_DIR", OUTPUT_DIR)
        load_model(model, weights_path)
        mapping_path = configs["evaluation"]["mapping_path"].replace("OUTPUT_DIR", OUTPUT_DIR)
        classes = load_yml(mapping_path).keys()
        evaluate_model(model, dataloader, classes, OUTPUT_DIR)
        LOGGER.info("Evaluation phase ended.")


if __name__ == "__main__":
    main()
