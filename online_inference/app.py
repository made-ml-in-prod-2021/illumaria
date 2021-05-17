import logging
import os
import pickle
import sys
from typing import List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

app = FastAPI()


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class HeartData(BaseModel):
    id: int = 0
    age: int = 18
    sex: int = 0
    cp: int = 0
    trestbps: int = 100
    chol: int = 245
    fbs: int = 0
    restecg: int = 0
    thalach: int = 100
    exang: int = 0
    oldpeak: float = 0
    slope: int = 0
    ca: int = 0
    thal: int = 0


class HeartResponse(BaseModel):
    id: int
    target: int


pipeline: Optional[Pipeline] = None


def make_prediction(
    data: List[HeartData],
    pipeline: Pipeline,
) -> List[HeartResponse]:
    data = pd.DataFrame(x.__dict__ for x in data)
    ids = [int(x) for x in data.id]
    predicts = pipeline.predict(data.drop("id", axis=1))

    return [
        HeartResponse(id=id_, target=int(target_))
        for id_, target_ in zip(ids, predicts)
    ]


@app.get("/")
def main():
    return "This is the entry point of our predictor."


@app.on_event("startup")
def load_model():
    model_path = os.getenv("PATH_TO_MODEL", default="model.pkl")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)
    global pipeline
    pipeline = load_object(model_path)


@app.get("/status")
def status() -> bool:
    return f"Pipeline is ready: {pipeline is not None}."


@app.api_route("/predict", response_model=List[HeartResponse], methods=["GET", "POST"])
def predict(request: List[HeartData]):
    return make_prediction(request, pipeline)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
