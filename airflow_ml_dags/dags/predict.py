from datetime import timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

AIRFLOW_RAW_DATA_PATH = "/opt/airflow/data/raw/{{ ds }}"
HOST_RAW_DATA_PATH = "/data/raw/{{ ds }}"
HOST_PROCESSED_DATA_PATH = "/data/processed/{{ ds }}"
HOST_PREDICTIONS_PATH = "/data/predictions/{{ ds }}"
HOST_DATA_DIR = Variable.get("HOST_DATA_DIR")

with DAG(
    "predict",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(1),
) as dag:
    wait_inference_data = FileSensor(
        task_id="wait-for-inference-data",
        filepath=f"{AIRFLOW_RAW_DATA_PATH}/data.csv",
        poke_interval=30
    )

    wait_model = FileSensor(
        task_id="wait-for-model",
        filepath="/opt/airflow/data/models/{{ var.value.model }}/model.pkl",
        poke_interval=30,
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input_dir {HOST_RAW_DATA_PATH} --output_dir {HOST_PROCESSED_DATA_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-preprocess-inference-data",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{HOST_DATA_DIR}:/data"],
    )

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--input_dir {HOST_PROCESSED_DATA_PATH} --output_dir {HOST_PREDICTIONS_PATH} "
                "--models_dir /data/models/{{ var.value.model }}",
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{HOST_DATA_DIR}:/data"],
    )

    [wait_inference_data, wait_model] >> preprocess >> predict
