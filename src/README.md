# Classifier-trains

[![PyPI version](https://badge.fury.io/py/classifier-trains.svg)](https://pypi.org/project/classifier-trains)
[![Python version](https://img.shields.io/pypi/pyversions/classifier-trains)](https://pypi.org/project/classifier-trains/)
[![Downloads](https://img.shields.io/pepy/dt/classifier-trains)](https://github.com/wyhwong/classifier-trains)
[![license](https://img.shields.io/badge/license-MIT-orange.svg)](https://github.com/wyhwong/classifier-trains/blob/main/LICENSE)
[![CI](https://github.com/wyhwong/classifier-trains/actions/workflows/main.yml/badge.svg)](https://github.com/wyhwong/classifier-trains/actions/workflows/main.yml/)

## Install as a package
```bash
pip3 install classifier-trains
```

## Examples of Pipeline Configuration

For example of pipeline configuration, please see [pipeline_config_only_train.yml](https://github.com/wyhwong/classifier-trains/blob/main/src/tests/test_data/pipeline_config_only_train.yml), [pipeline_config_only_eval.yml](https://github.com/wyhwong/classifier-trains/blob/main/src/tests/test_data/pipeline_config_only_eval.yml), [full_pipeline_config.yml](https://github.com/wyhwong/classifier-trains/blob/main/src/tests/test_data/full_pipeline_config.yml).

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

## Pipeline Parameter Table

#### Pipeline Parameters

| Parameter | Description | Type | Default | Choices |
| --- | --- | --- | --- | --- |
| enable_training | Enable training | bool | False | True, False |
| enable_evaluation | Enable evaluation | bool | False | True, False |

#### Model Parameters

| Parameter | Description | Type | Default | Choices |
| --- | --- | --- | --- | --- |
| model | Model architecture to use | str | / | resnet18, resnet34, resnet50, resnet152, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn, squeezenet1_0, squeezenet1_1, densenet121, densenet161, densenet169, densenet201, inception_v3 |
| num_classes | Number of classes | int | / | Any positive integer |
| weights | Pretrained weights | str | DEFAULT | Check PyTorch Documentation |
| checkpoint_path | Path to checkpoint | Optional[str] | None | Self defined path |

#### Dataloader Parameters

| Parameter | Description | Type | Default | Choices |
| --- | --- | --- | --- | --- |
| batch_size | Batch size | int | / | Any positive integer |
| num_workers | Number of workers | int | 0 | Any non negative integer |

#### Resize Parameters

| Parameter | Description | Type | Default | Choices |
| --- | --- | --- | --- | --- |
| width | Width of resized image | int | / | Any positive integer |
| height | Height of resized image | int | / | Any positive integer |
| interpolation | Interpolation method | str | bicubic | nearest, nearest_exact, bilinear, bicubic |
| padding | Padding | Optional[str] | None | top_left, top_right, bottom_left, bottom_right, center |
| maintain_aspect_ratio | Maintain aspect ratio | bool | False | True, False |

#### Spatial Transform Parameters

| Parameter | Description | Type | Default | Choices |
| --- | --- | --- | --- | --- |
| hflip_prob | Horizontal flip probability | float | / | Any float between 0 and 1 |
| vflip_prob | Vertical flip probability | float | / | Any float between 0 and 1 |
| max_rotate_in_degree | Maximum rotation in degree | float | 0.0 | Any non negative float |
| allow_center_crop | Allow center crop | bool | False | True, False |
| allow_random_crop | Allow random crop | bool | False | True, False |

#### Color Transform Parameters

| Parameter | Description | Type | Default | Choices |
| --- | --- | --- | --- | --- |
| allow_gray_scale | Allow gray scale | bool | False | True, False |
| allow_random_color | Allow random color | bool | False | True, False |

#### Preprocessing Parameters

| Parameter | Description | Type | Default | Choices |
| --- | --- | --- | --- | --- |
| mean | Mean of dataset | List[float] | / | Any list of floats |
| std | Standard deviation of dataset | List[float] | / | Any list of floats |

#### Optimizer Parameters

| Parameter | Description | Type | Default | Choices |
| --- | --- | --- | --- | --- |
| name | Optimizer to use | str | / | sgd, adam, rmsprop, adamw |
| lr | Learning rate | float | / | Any positive float |
| weight_decay | Weight decay | float | 0.0 | Any non negative float |
| momentum | Momentum | Optional[float] | None | Any non negative float |
| alpha | Alpha | Optional[float] | None | Any non negative float |
| betas | Betas | Optional[Tuple[float, float]] | None | Any tuple of non negative floats |

#### Scheduler Parameters

| Parameter | Description | Type | Default | Choices |
| --- | --- | --- | --- | --- |
| name | Scheduler to use | str | / | step, consine |
| lr_min | Minimum learning rate | Optional[float] | None | Any non negative float |
| step_size | Step size | Optional[int] | None | Any positive integer |
| gamma | Gamma | Optional[float] | None | Any non negative float |

#### Training Parameters

| Parameter | Description | Type | Default | Choices |
| --- | --- | --- | --- | --- |
| name | Name of the experiment | str | / | Any string |
| num_epochs | Number of epochs | int | / | Any positive integer |
| trainset_dir | Path to training dataset | str | / | Any string |
| valset_dir | Path to validation dataset | str | / | Any string |
| testset_dir | Path to testing dataset | Optional[str] | None | Any string |
| device | Device to use | str | cuda | cpu, cuda, etc. |
| max_num_hrs | Maximum number of hours to run | Optional[float] | None | Any non negative float |
| criterion | What criterion to use for best model | str | loss | loss, accuracy |
| validate_every | Validate every n epochs | int | 1 | Any positive integer |
| save_every | Save model every n epochs | int | 3 | Any positive integer |
| patience | Patience for early stopping | int | 5 | Any positive integer |
| random_seed | Random seed | int | 42 | Any positive integer |
| precision | Precision | int | 64 | 16, 32, 64 |
| export_last_as_onnx | Export last model as ONNX | bool | False | True, False |
| export_best_as_onnx | Export best model as ONNX | bool | False | True, False |

#### Evaluation Parameters

| Parameter | Description | Type | Default | Choices |
| --- | --- | --- | --- | --- |
| name | Name of the experiment | str | / | Any string |
| evalset_dir | Path to evaluation dataset | str | / | Any string |
| device | Device to use | str | cuda | cpu, cuda, etc. |
| precision | Precision | int | 64 | 16, 32, 64 |
| random_seed | Random seed | int | 42 | Any positive integer |
