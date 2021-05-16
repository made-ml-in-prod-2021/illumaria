# ML project for ML in Production course

## Prerequisites

* Python >= 3.7
* pip >= 19.0.3

## Installation

```bash
git clone https://github.com/made-ml-in-prod-2021/illumaria.git
git checkout homework2
cd online_inference
docker build -t illumaria/online_inference:v1 .
```

## Usage

### Predict with model

```bash
docker run -p 8000:8000 illumaria/online_inference:v1
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

## Project structure

Work in progress.
