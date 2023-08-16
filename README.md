# TAO-like Classifier training pipeline using PyTorch (TCP)
This repository aims to provide a primitive tool to finetune state-of-the-art models with PyTorch implementation, similar to Nvidia TAO but with more flexibility in augmentation and models. Like TAO classification, all parameters are configurable in .yml config to minimize the need of modifying scripts.

## Description and Usage
Get config ready, get dataset ready, no coding (hopefully :D), start PyTorch training.

```bash
# Build Docker image
make build

# Modify configs/train.yml according to your needs
# Then start training
DATASET=<path to image dataset> CONFIG=<path to yml config> OUTPUT_DIR=<path to output folder> make train

# Example
DATASET=${PWD}/dataset CONFIG=${PWD}/configs/train.yml OUTPUT_DIR=${PWD}/results make train
```

## Folder Structure for Dataset
```
Dataset Directory
├── train
│   ├── <class1>
│   │   ├── <image1>
│   │   └── ...
│   └── <class2>
│       ├── <image1>
│       └── ...
├── val
│   ├── <class1>
│   │   ├── <image1>
│   │   └── ...
│   └── <class2>
│       ├── <image1>
│       └── ...
└── eval
    ├── <class1>
    │   ├── <image1>
    │   └── ...
    └── <class2>
        ├── <image1>
        └── ...
```

## After Training

By default, the pipeline will output the following:

1. Weights of best and last model
2. yml file of class mapping
3. Preview of trainset and valset during training
4. Line plots of accuracy history and loss history
5. csv files of accuracy history and loss history
6. ROC curves of model in each class
7. Full training log

## To Be Developed

1. Upgrade to PyTorch 2
2. Implement detector training

## Author
[@wyhwong](https://github.com/wyhwong)
