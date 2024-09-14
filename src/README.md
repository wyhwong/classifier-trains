# Classifier-trains

For project details, please see [README.md](https://github.com/wyhwong/classifier-trains)

## Install as a package
```bash
pip3 install classifier-trains
```

## Makefile Commands (Only when cloned from GitHub)

```bash
# Check all available commands
make help
```

## After installation

For example of pipeline configuration, please see [pipeline_setting.yml](https://github.com/wyhwong/classifier-trains/blob/main/src/pipeline_setting.yml).

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
