import json

import pytest
from fastapi.testclient import TestClient

from app import app, load_model
from src.entities import HeartData


@pytest.fixture(scope="session", autouse=True)
def initialize_model():
    load_model()


@pytest.fixture()
def test_data():
    data = [HeartData(id=0), HeartData(id=1)]
    return data


def test_main_endpoint_works_correctly():
    with TestClient(app) as client:
        response = client.get("/")
        assert 200 == response.status_code


def test_status_endpoint_works_correctly():
    with TestClient(app) as client:
        expected_response = "Pipeline is ready: True."
        response = client.get("/status")
        assert 200 == response.status_code
        assert expected_response == response.json()


def test_predict_endpoint_works_correctly(test_data):
    with TestClient(app) as client:
        response = client.post(
            "/predict", data=json.dumps([x.__dict__ for x in test_data])
        )
        assert 200 == response.status_code
        assert len(response.json()) == len(test_data)
        assert 0 == response.json()[0]["id"]
        assert response.json()[0]["target"] in [0, 1]


def test_predict_endpoint_with_wrong_data_type():
    with TestClient(app) as client:
        data = HeartData()
        data.age = "old"
        response = client.post("/predict", data=json.dumps([data.__dict__]))
        expected_message = "value is not a valid integer"
        assert 422 == response.status_code
        assert expected_message == response.json()["detail"][0]["msg"]


def test_validation_for_binary_values():
    with TestClient(app) as client:
        data = HeartData()
        data.sex = 42
        response = client.post("/predict", data=json.dumps([data.__dict__]))
        assert 400 == response.status_code
        expected_message = "Parameter 'sex' has value 42 which is not binary."
        assert (expected_message == response.json()["detail"])


def test_validation_for_out_of_range_values():
    with TestClient(app) as client:
        data = HeartData()
        data.age = 1000
        response = client.post("/predict", data=json.dumps([data.__dict__]))
        assert 400 == response.status_code
        expected_message = "Parameter 'age' has value 1000 which is out of [0, 150] range."
        assert (expected_message == response.json()["detail"])
