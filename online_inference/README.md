# ML project for ML in Production course

## Prerequisites

* Python >= 3.7
* pip >= 19.0.3
* docker >= 4.4.4

## Installation

### From source

```bash
git clone https://github.com/made-ml-in-prod-2021/illumaria.git
cd illumaria
git checkout homework2
cd online_inference
docker build -t illumaria/online_inference:v3 .
```

### From DockerHub

```bash
docker pull illumaria/online_inference:v3
```

## Usage

### Run inference

```bash
docker run --rm -p 8000:8000 illumaria/online_inference:v3
python make_request.py
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

## Docker optimizations

The optimizations included:
* listing only the required packages in `requirements.txt`;
* copying only the required source files to Docker image (no data, tests, etc.);
* trying out [different tags](https://hub.docker.com/r/illumaria/online_inference/tags?page=1&ordering=last_updated) for Dockerfile:
    * **v1**: python:3.7 (501 MB compressed)
    * **v2**: python:3.7-slim (219.99 MB compressed)
    * **v3**: python:3.7-slim-stretch (215.28 MB compressed)

## Project structure

------------

    ├── data
    │   └── inference_test             <- The data needed to make test requests.
    │
    ├── src                            <- Source code for use in this project.
    │   ├── __init__.py                <- Makes src a Python module.
    │   │
    │   ├── entities                   <- Parameters for different project modules.
    │   │   ├── heart_data.py
    │   │   └── heart_response.py    
    │   │
    │   └── validate.py
    │
    ├── tests                          <- Code to test project modules and pipelines.
    │   └── test_app.py
    │
    ├── app.py                         <- FastAPI application code.
    │
    ├── Dockerfile                     <- Docker image building file.
    │
    ├── make_request.py                <- Script to make requests to application
    │                                     predict endpoint.
    │
    ├── model.pkl                      <- Pretrained transformer and model dump.
    │
    ├── README.md                      <- The README for developers using this project.
    │
    └── requirements.txt               <- The requirements file for reproducing the environment.

------------
