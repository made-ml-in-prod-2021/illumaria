# ML project for ML in Production course

## Prerequisites

* Python >= 3.6
* pip >= 20.0.0
* docker
* [docker-compose](https://docs.docker.com/compose/install/) >= 1.25.0

## Installation

```bash
git clone https://github.com/made-ml-in-prod-2021/illumaria.git
cd illumaria
git checkout homework3
cd airflow_ml_dags
```

## Usage

```bash
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
export HOST_DATA_DIR=$(pwd)/data
docker-compose up --build
```

In those cases when docker only runs with `sudo`, don't forget that `sudo` has its own environment variables, so the last command will look differently:

```bash
sudo -E docker-compose up --build
```

### Run linter

```bash
flake8 . --count --max-line-length=120 --statistics
```
