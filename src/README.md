# Classifier-trains

## Makefile Commands

```bash
# Check all available commands
make help
```

## After installation

For example of pipeline configuration, please see [pipeline_setting.yml](./pipeline_setting.yml).

```bash
# Run training or evaluation
python3 -m classifier_trains run --config <path to yml config> --output_dir <output_dir>

# Run training or evaluation with profiling, which will generate a profile report
python3 -m classifier_trains profile --config <path to yml config> --output_dir <output_dir>

# Compute mean and std of dataset
python3 -m classifier_trains compute-mean-and-std --dir-path <path to dataset>

# Get output mapping of dataset
python3 -m classifier_trains get-output-mapping --dir-path <path to dataset>
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

![ROC curve in tensorboard](../docs/preview-tensorboard-1.png)

![Sample images and parameters](../docs/preview-tensorboard-2.png)

## Profile Report

If you run the training with profiling, a profile report will be generated in the output directory. You can see the time spent on each function. It gives you a better understanding of the performance of the training process, and a sense of where to optimize.

![Profile report](../docs/preview-profile-report.png)
