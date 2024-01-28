# TAO-like PyTorch Object Classifier Training Pipeline

## Run in Docker Environment

```bash
# Build Docker image
make build

# Modify setting.yml according to your needs
# Then start training
DATA_DIR=< path to image dataset > CONFIG_PATH=< path to yml config > OUTPUT_DIR=< path to output folder > make train

# Example (also default values)
DATA_DIR=./dataset CONFIG_PATH=./train.yml OUTPUT_DIR=./results make train
```

## Run with Poetry Environment

```bash
# Install dependencies in Poetry
make local

# Modify setting.yml according to your needs
# Put dataset in ./dataset

# Then start training
cd src
poetry run python3 main.py
```

## GNU Make Commands for Development

```bash
# Install dependencies in Poetry
make local

# Run static code analysis
# Components included:
#   - black (formatter)
#   - bandit (security linter)
#   - pylint (linter)
#   - mypy (type checker)
#   - isort (import sorter)
make analyze

# Update dependencies in Poetry
make update

# After developement
make format
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
