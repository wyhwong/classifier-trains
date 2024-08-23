# TAO-like PyTorch Object Classifier Training Pipeline

## Makefile Commands

```bash
# Check all available commands
make help
```

## After installation

For example of pipeline configuration, please see [pipeline_setting.yml](./pipeline_setting.yml).

```bash
# Run training or evaluation
python3 -m pipeline run --config <path to yml config> --output_dir <output_dir>

# Compute mean and std of dataset
python3 -m pipeline compute-mean-and-std --dir-path <path to dataset>

# Get output mapping of dataset
python3 -m pipeline get-output-mapping --dir-path <path to dataset>
```

## Expected Folder Structure for Dataset
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

The logs and checkpoints will be saved in the output directory, and logs are in tensorboard format. In tensorboard, you will be able to see the ROC curve, sample images in training, parameters like learning rate and momentum, and metrics like accuracy and loss.

```bash
# Run tensorboard
tensorboard --logdir <output_dir>
```

![ROC curve in tensorboard](../docs/tensorboard_1.png)

![Sample images and parameters](../docs/tensorboard_2.png)
