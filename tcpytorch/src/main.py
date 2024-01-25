#!/usr/bin/env python3
import numpy as np
import torch
import utils
from torchvision import datasets


SETUP = utils.common.get_config()["setup"]
SEED = SETUP["seed"]
TRAIN = SETUP["enable_training"]
EVAL = SETUP["enable_evaluation"]
EXPORT = SETUP["enable_export"]
LOGGER = utils.logger.get_logger("main")
utils.common.check_and_create_dir(utils.env.OUTPUT_DIR)
torch.manual_seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = f"/results/{int(len(glob('/results/*'))+1)}_{LABEL}"
os.mkdir(OUTPUT_DIR)


def main():
    configs = utils.common.get_config()
    utils.common.save_dict_as_yml(f"{utils.env.OUTPUT_DIR}/train.yml", configs)
    LOGGER.info("Initializing training using %s", utils.env.DEVICE)
    if TRAIN or EVAL:
        data_tranforms = utils.preprocessing.get_transforms(**configs["preprocessing"])

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
        utils.training.get_class_mapping(image_datasets["train"], f"{utils.env.OUTPUT_DIR}/classMapping.yml")
        for phase in ["train", "val"]:
            utils.visualization.get_dataset_preview(
                dataset=image_datasets[phase],
                mean=np.array(configs["preprocessing"]["mean"]),
                std=np.array(configs["preprocessing"]["std"]),
                filename_remark=phase,
                output_dir=utils.env.OUTPUT_DIR,
            )
        criterion = torch.nn.CrossEntropyLoss()
        model = utils.model.initialize_model(**configs["model"])
        optimizer = utils.training.get_optimizer(params=model.parameters(), **configs["training"]["optimizer"])
        scheduler = utils.training.get_scheduler(
            optimizer=optimizer,
            num_epochs=configs["training"]["train_model"]["num_epochs"],
            **configs["training"]["scheduler"],
        )
        LOGGER.info("Loaded all parameters, training starts.")
        (
            model,
            best_weights,
            last_weights,
            train_loss,
            train_acc,
        ) = utils.training.train_model(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **configs["training"]["train_model"],
        )
        LOGGER.info("Training ended, visualizing results.")
        utils.visualization.visualize_acc_and_loss(
            train_loss=train_loss, train_acc=train_acc, output_dir=utils.env.OUTPUT_DIR
        )
        LOGGER.info("Training phase ended.")

    if EXPORT:
        LOGGER.info("Starting phase: Export.")
        if configs["export"]["save_last_weight"]:
            utils.model.save_weights(last_weights, f"{utils.env.OUTPUT_DIR}/lastModel.pt")
        if configs["export"]["save_best_weight"]:
            utils.model.save_weights(best_weights, f"{utils.env.OUTPUT_DIR}/bestModel.pt")
        if configs["export"]["export_last_weight"]:
            model.load_state_dict(last_weights)
            utils.export.export_model_to_onnx(
                model=model,
                input_height=configs["preprocessing"]["height"],
                input_width=configs["preprocessing"]["width"],
                export_path=f"{utils.env.OUTPUT_DIR}/lastModel.onnx",
            )
            utils.export.check_model_is_valid(model_path=f"{utils.env.OUTPUT_DIR}/lastModel.onnx")
        if configs["export"]["export_best_weight"]:
            model.load_state_dict(best_weights)
            utils.export.export_model_to_onnx(
                model=model,
                input_height=configs["preprocessing"]["height"],
                input_width=configs["preprocessing"]["width"],
                export_path=f"{utils.env.OUTPUT_DIR}/bestModel.onnx",
            )
            utils.export.check_model_is_valid(f"{utils.env.OUTPUT_DIR}/bestModel.onnx")
        LOGGER.info("Export phase ended.")

    if EVAL:
        LOGGER.info("Starting phase: Evaluation.")
        eval_configs = configs["evaluation"]
        image_dataset = datasets.ImageFolder(eval_configs["evalset_dir"], data_tranforms["val"])
        dataloader = torch.utils.data.DataLoader(
            image_dataset,
            batch_size=configs["dataset"]["batch_size"],
            shuffle=False,
            num_workers=configs["dataset"]["num_workers"],
        )
        model = utils.model.initialize_model(**configs["model"])
        weights_path = eval_configs["weights_path"].replace("OUTPUT_DIR", utils.env.OUTPUT_DIR)
        utils.model.load_model(model, weights_path)
        mapping_path = eval_configs["mapping_path"].replace("OUTPUT_DIR", utils.env.OUTPUT_DIR)
        classes = utils.common.load_yml(mapping_path).keys()
        utils.evaluation.evaluate_model(model, dataloader, classes, utils.env.OUTPUT_DIR)
        LOGGER.info("Evaluation phase ended.")

        if eval_configs["roc_curve"]:
            for model_config in eval_configs["roc_curve"]:
                model_config["path"] = model_config["path"].replace("OUTPUT_DIR", utils.env.OUTPUT_DIR)
            utils.evaluation.visualize_roc_curve(eval_configs["roc_curve"], dataloader, classes, utils.env.OUTPUT_DIR)


if __name__ == "__main__":
    main()
