#!/usr/bin/env python3
import torch
import numpy as np

from utils.common import getConfig, getLogger, saveConfig, checkAndCreateDir
from utils.model import initializeModel, unfreezeAllParams
from utils.preprocessing import getTransforms
from utils.training import train_model, getOptimizer, getScheduler, getPreview
from utils.export import exportModel
from utils.evaluation import evaluateModel
from utils.visualization import initializePlot

SEED = getConfig()["seed"]
LOGGER = getLogger("Main")
torch.manual_seed(SEED)
np.random.seed(SEED)

def main():
    pass

if __name__ == "__main__":
    main()
