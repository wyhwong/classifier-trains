#!/usr/bin/env python3
import torch
import numpy as np
from torchvision import datasets

from utils.common import get_config, save_dict_as_yml, check_and_create_dir, load_yml
from utils.logger import get_logger
from utils.model import initialize_model, load_model, save_weights
from utils.preprocessing import get_transforms
from utils.training import train_model, get_optimizer, get_scheduler, get_class_mapping
from utils.export import export_model_to_onnx, check_model_is_valid
from utils.evaluation import evaluate_model
from utils.visualization import visualize_acc_and_loss, get_dataset_preview

SETUP = get_config()["setup"]
SEED = SETUP["seed"]
OUTPUTDIR = SETUP["outputDir"]
TRAIN = SETUP["enableTraining"]
EVAL = SETUP["enableEvaluation"]
EXPORT = SETUP["enableExport"]
LOGGER = get_logger("Main")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

check_and_create_dir(OUTPUTDIR)
torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    configs = get_config()
    save_dict_as_yml(f"{OUTPUTDIR}/train.yml", configs)
    LOGGER.info(f"Initializing training using {DEVICE=}")
    if TRAIN or EVAL:
        data_tranforms = get_transforms(**configs["preprocessing"])

    if TRAIN:
        LOGGER.info("Starting phase: Training, loading necessary parameters.")
        image_datasets = {
            x: datasets.ImageFolder(configs["dataset"][f"{x}setDir"], data_tranforms[x]) for x in ["train", "val"]
        }
        dataloaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x],
                batch_size=configs["dataset"]["batchSize"],
                shuffle=True,
                num_workers=configs["dataset"]["numWorkers"],
            )
            for x in ["train", "val"]
        }
        get_class_mapping(image_datasets["train"], f"{OUTPUTDIR}/classMapping.yml")
        for phase in ["train", "val"]:
            get_dataset_preview(
                dataset=image_datasets[phase],
                mean=np.array(configs["preprocessing"]["mean"]),
                std=np.array(configs["preprocessing"]["std"]),
                filename_remark=phase,
                output_dir=OUTPUTDIR,
            )
        criterion = torch.nn.CrossEntropyLoss()
        model = initialize_model(**configs["model"])
        optimizer = get_optimizer(params=model.parameters(), **configs["training"]["optimizer"])
        scheduler = get_scheduler(
            optimizer=optimizer,
            num_epochs=configs["training"]["trainModel"]["numEpochs"],
            **configs["training"]["scheduler"],
        )
        LOGGER.info("Loaded all parameters, training starts.")
        model, best_weights, last_weights, train_loss, train_acc = train_model(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **configs["training"]["trainModel"],
        )
        LOGGER.info("Training ended, visualizing results.")
        visualize_acc_and_loss(train_loss=train_loss, train_acc=train_acc, output_dir=OUTPUTDIR)
        LOGGER.info("Training phase ended.")

    if EXPORT:
        LOGGER.info("Starting phase: Export.")
        if configs["export"]["saveLastWeight"]:
            save_weights(last_weights, f"{OUTPUTDIR}/lastModel.pt")
        if configs["export"]["saveBestWeight"]:
            save_weights(best_weights, f"{OUTPUTDIR}/bestModel.pt")
        if configs["export"]["exportLastWeight"]:
            model.load_state_dict(last_weights)
            export_model_to_onnx(
                model=model,
                input_height=configs["preprocessing"]["height"],
                input_width=configs["preprocessing"]["width"],
                export_path=f"{OUTPUTDIR}/lastModel.onnx",
            )
            check_model_is_valid(model_path=f"{OUTPUTDIR}/lastModel.onnx")
        if configs["export"]["exportBestWeight"]:
            model.load_state_dict(best_weights)
            export_model_to_onnx(
                model=model,
                input_height=configs["preprocessing"]["height"],
                input_width=configs["preprocessing"]["width"],
                export_path=f"{OUTPUTDIR}/bestModel.onnx",
            )
            check_model_is_valid(f"{OUTPUTDIR}/bestModel.onnx")
        LOGGER.info("Export phase ended.")

    if EVAL:
        LOGGER.info("Starting phase: Evaluation.")
        image_dataset = datasets.ImageFolder(configs["evaluation"][f"evalsetDir"], data_tranforms["val"])
        dataloader = torch.utils.data.DataLoader(
            image_dataset,
            batch_size=configs["dataset"]["batchSize"],
            shuffle=False,
            num_workers=configs["dataset"]["numWorkers"],
        )
        model = initialize_model(**configs["model"])
        load_model(model, configs["evaluation"]["modelPath"])
        classes = load_yml(configs["evaluation"]["mappingPath"]).keys()
        evaluate_model(model, dataloader, classes, OUTPUTDIR)
        LOGGER.info("Evaluation phase ended.")


if __name__ == "__main__":
    main()
