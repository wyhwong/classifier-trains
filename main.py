#!/usr/bin/env python3
import torch
import numpy as np

from utils.common import getConfig, getLogger, saveConfig, checkAndCreateDir
from utils.model import initializeModel, loadModel
from utils.preprocessing import getTransforms
from utils.training import trainModel, getOptimizer, getScheduler, getPreview
from utils.export import exportModelWeight, exportModelToONNX
from utils.evaluation import evaluateModel
from utils.visualization import visualizeAccAndLoss

SETUP = getConfig()["setup"]
SEED = SETUP["seed"]
OUTPUTDIR = SETUP["outputDir"]
checkAndCreateDir(OUTPUTDIR)
TRAIN = SETUP["enableTraining"]
EVAL = SETUP["enableEvaluation"]
EXPORT = SETUP["enableExport"]
LOGGER = getLogger("Main")

torch.manual_seed(SEED)
np.random.seed(SEED)

def main():
    config = getConfig()
    saveConfig(path=f"{OUTPUTDIR}/config.yml", config=config)
    if TRAIN or EVAL:
        dataTranforms = getTransforms(**config["preprocessing"])

    if TRAIN:
        LOGGER.info("Starting phase: Training, loading necessary parameters")
        dataloaders = ""
        criterion = ""
        model = initializeModel(**config["model"])
        optimizer = getOptimizer(**config["training"]["optimizer"])
        scheduler = getScheduler(**config["training"]["scheduler"])
        getPreview("")
        LOGGER.info("Loaded all parameters, training starts.")
        bestModel, lastModel, trainLoss, trainAcc = trainModel(model=model,
                                                            dataloaders=dataloaders,
                                                            criterion=criterion,
                                                            optimizier=optimizer,
                                                            scheduler=scheduler,
                                                            **config["training"]["trainModel"])
        LOGGER.info("Training ended, visualizing results")
        visualizeAccAndLoss(trainLoss, trainAcc)
        LOGGER.info("Training phase ended.")

    if EXPORT:
        LOGGER.info("Starting phase: Export")
        if config["export"]["saveLastWeight"]:
            exportModelWeight(model=lastModel, exportPath=f"{OUTPUTDIR}/lastModel.pt")
        if config["export"]["saveBestWeight"]:
            exportModelWeight(model=bestModel, exportPath=f"{OUTPUTDIR}/bestModel.pt")
        if config["export"]["exportLastWeight"]:
            exportModelToONNX(model=lastModel, exportPath=f"{OUTPUTDIR}/lastModel.onnx")
        if config["export"]["exportBestWeight"]:
            exportModelToONNX(model=bestModel, exportPath=f"{OUTPUTDIR}/bestModel.onnx") 
        LOGGER.info("Export phase ended")

    if EVAL:
        LOGGER.info("Starting phase: Evaluation")
        dataloader = ""
        model = loadModel(modelPath=config["evaluation"]["modelPath"])
        evaluateModel(model=model, dataloader=dataloader,
                      resultsDir=f"{OUTPUTDIR}/modelEval")
        LOGGER.info("Evaluation phase ended")


if __name__ == "__main__":
    main()
