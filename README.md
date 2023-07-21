# TAO-like Classifier PyTorch-training (TCP)
This repository aims to provide a primitive tool to finetune state-of-the-art models with PyTorch implementation, similar to Nvidia TAO but with more flexibility in augmentation and models. Like TAO classification, all parameters are configurable in .yml config to minimize the need of modifying scripts.

## Description and Usage
Get config ready, get dataset ready, no coding (hopefully :D), start PyTorch training.

```bash
# Build Docker image
make build

# Modify config/config.yml according to your needs
# Then start training
dataset=<path to image dataset> config=<path to yml config> outputDir=<path to output folder> make train

# Example
dataset=${PWD}/dataset config=${PWD}/configs/train.yml outputDir=${PWD}/results make train
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

Note: Evaluation function is not ready.

## Author
[@wyhwong](https://github.com/wyhwong)
