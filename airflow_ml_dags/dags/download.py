from datetime import timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago


HOST_DATA_DIR = Variable.get("HOST_DATA_DIR")

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "download",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(14),
) as dag:
    download = DockerOperator(
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{HOST_DATA_DIR}:/data"]
    )

    download
