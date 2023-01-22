#!/usr/bin/env python3
import torch
import numpy as np
from torchvision import datasets

from utils.common import getConfig, getLogger, saveConfig, checkAndCreateDir
from utils.model import initializeModel, loadModel, saveWeights
from utils.preprocessing import getTransforms
from utils.training import trainModel, getOptimizer, getScheduler, getPreview
from utils.export import exportModelToONNX, checkModelIsValid
from utils.evaluation import evaluateModel
from utils.visualization import visualizeAccAndLoss

SETUP = getConfig()["setup"]
SEED = SETUP["seed"]
OUTPUTDIR = SETUP["outputDir"]
TRAIN = SETUP["enableTraining"]
EVAL = SETUP["enableEvaluation"]
EXPORT = SETUP["enableExport"]
LOGGER = getLogger("Main")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


checkAndCreateDir(OUTPUTDIR)
torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    config = getConfig()
    saveConfig(path=f"{OUTPUTDIR}/config.yml", config=config)
    LOGGER.info(f"Initializing training using {DEVICE=}")
    if TRAIN or EVAL:
        dataTranforms = getTransforms(**config["preprocessing"])

    if TRAIN:
        LOGGER.info("Starting phase: Training, loading necessary parameters.")
        imageDatasets = {x: datasets.ImageFolder(config["dataset"][f"{x}setDir"],
                                                 dataTranforms[x]) for x in ["train", "val"]}
        dataloaders = {x: torch.utils.data.DataLoader(imageDatasets[x],
                                                      batch_size=config["dataset"]["batchSize"],
                                                      shuffle=True,
                                                      num_workers=config["dataset"]["numWorkers"]) for x in ["train", "val"]}
        criterion = torch.nn.CrossEntropyLoss()
        model = initializeModel(**config["model"])
        optimizer = getOptimizer(params=model.parameters(),
                                 **config["training"]["optimizer"])
        scheduler = getScheduler(optimizer=optimizer,
                                 numEpochs=config["training"]["trainModel"]["numEpochs"],
                                 **config["training"]["scheduler"])
        # getPreview()
        LOGGER.info("Loaded all parameters, training starts.")
        model, bestWeights, lastWeights, trainLoss, trainAcc = trainModel(model=model,
                                                                          dataloaders=dataloaders,
                                                                          criterion=criterion,
                                                                          optimizer=optimizer,
                                                                          scheduler=scheduler,
                                                                          **config["training"]["trainModel"])
        LOGGER.info("Training ended, visualizing results.")
        visualizeAccAndLoss(trainLoss=trainLoss, trainAcc=trainAcc, outputDir=OUTPUTDIR)
        LOGGER.info("Training phase ended.")

    if EXPORT:
        LOGGER.info("Starting phase: Export.")
        if config["export"]["saveLastWeight"]:
            saveWeights(weights=lastWeights, exportPath=f"{OUTPUTDIR}/lastModel.pt")
        if config["export"]["saveBestWeight"]:
            saveWeights(weights=bestWeights, exportPath=f"{OUTPUTDIR}/bestModel.pt")
        if config["export"]["exportLastWeight"]:
            model.load_state_dict(lastWeights)
            exportModelToONNX(model=model,
                              height=config["preprocessing"]["height"],
                              width=config["preprocessing"]["width"],
                              exportPath=f"{OUTPUTDIR}/lastModel.onnx")
            checkModelIsValid(modelPath=f"{OUTPUTDIR}/lastModel.onnx")
        if config["export"]["exportBestWeight"]:
            model.load_state_dict(bestWeights)
            exportModelToONNX(model=model,
                              height=config["preprocessing"]["height"],
                              width=config["preprocessing"]["width"],
                              exportPath=f"{OUTPUTDIR}/bestModel.onnx")
            checkModelIsValid(modelPath=f"{OUTPUTDIR}/bestModel.onnx")
        LOGGER.info("Export phase ended.")

    if EVAL:
        LOGGER.info("Starting phase: Evaluation.")
        imageDataset = datasets.ImageFolder(config["evaluation"][f"evalsetDir"],
                                            dataTranforms["val"])
        dataloader = torch.utils.data.DataLoader(imageDataset,
                                                  batch_size=config["dataset"]["batchSize"],
                                                  shuffle=False,
                                                  num_workers=config["dataset"]["numWorkers"])
        model = initializeModel(**config["model"])
        loadModel(model=model, modelPath=config["evaluation"]["modelPath"])
        evaluateModel(model=model, dataloader=dataloader,
                      resultsDir=f"{OUTPUTDIR}/modelEval")
        LOGGER.info("Evaluation phase ended.")


if __name__ == "__main__":
    main()
