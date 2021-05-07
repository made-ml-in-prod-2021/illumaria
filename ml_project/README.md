# ML project for ML in Production course

## Prerequisites

* Python >= 3.7
* pip >= 19.0.3
* [Heart Disease UCI Dataset](https://www.kaggle.com/ronitf/heart-disease-uci)

## Installation

```bash
git clone https://github.com/made-ml-in-prod-2021/illumaria.git
git checkout homework1
cd ml_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Build EDA report

```bash
python src/visualization/visualize.py -i data/raw/heart.csv -o reports/heart
```

### Train model

```bash
python src/train_pipeline.py [--config-path <path>] [--config-name <name>]
```

### Predict with model

```bash
python src/predict_pipeline.py [--config-path <path>] [--config-name <name>]
```

### Run tests

```bash
pip install pytest pytest-cov
python -m pytest . -v --cov
```

### Run linter

```bash
flake8 . --count --max-line-length=120 --statistics
```

## Project structure
------------

    ├── configs                        <- Configuration files for project modules.
    │   ├── feature_params
    │   │   └── default.yaml
    │   │
    │   ├── split_params
    │   │   ├── val_10_rand_42.yaml
    │   │   └── val_20_rand_4.yaml
    │   │
    │   ├── train_params
    │   │   ├── logistic_regression.yaml
    │   │   └── random_forest_classifier.yaml
    │   │
    │   ├── predict_config.yaml
    │   └── train_config.yaml
    │
    ├── data
    │   └── raw                        <- The original, immutable data dump.
    │
    ├── logs                           <- Training and prediction log files.
    │
    ├── models                         <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks                      <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                     the creator's initials, and a short `-` delimited description, e.g.
    │                                     `1.0-jqp-initial-data-exploration`.
    │
    ├── reports                        <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── src                            <- Source code for use in this project.
    │   ├── __init__.py                <- Makes src a Python module.
    │   │
    │   ├── data                       <- Scripts to download or generate data.
    │   │   └── make_dataset.py
    │   │
    │   ├── entities                   <- Parameters for different project modules.
    │   │   ├── feature_params.py
    │   │   ├── predict_pipeline_params.py
    │   │   ├── project_params.py
    │   │   ├── split_params.py
    │   │   ├── train_params.py
    │   │   └── train_pipeline_params.py    
    │   │
    │   ├── features                   <- Scripts to turn raw data into features for modeling.
    │   │   └── build_features.py
    │   │
    │   ├── models                     <- Scripts to train models and then use trained models to make
    │   │   │                             predictions.
    │   │   ├── encoders.py
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization              <- Scripts to create exploratory and results oriented visualizations.
    │   │   └── visualize.py
    │   │
    │   ├── predict_pipeline.py
    │   └── train_pipeline.py
    │
    ├── tests                          <- Code to test project modules and pipelines.
    │   ├── data
    │   │   └── test_make_dataset.py
    │   │
    │   ├── features
    │   │   ├── test_make_categorical_features.py
    │   │   └── test_make_features.py
    │   │
    │   ├── models
    │   │   ├── test_encoders.py
    │   │   └── test_train_model.py
    │   │
    │   ├── conftest.py
    │   ├── test_end2end_training.py
    │   └── train_data_sample.csv
    │
    ├── LICENSE
    │
    ├── README.md                      <- The top-level README for developers using this project.
    │
    ├── requirements.txt               <- The requirements file for reproducing the analysis environment, e.g.
    │                                     generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                       <- Makes project pip installable (pip install -e .) so src can be imported.
    │
    └── tox.ini                        <- tox file with settings for running tox; see tox.readthedocs.io.


------------
