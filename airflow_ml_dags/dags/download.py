import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago


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
    start_date=days_ago(1),
) as dag:
    download = DockerOperator(
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        auto_remove=True,
        # volumes=[f'{os.environ["HOST_DATA_DIR"]}:/data']
        volumes=[f'/home/uadmin/Projects/haystack/ml_in_prod/illumaria/airflow_ml_dags/data:/data']
    )

    download
