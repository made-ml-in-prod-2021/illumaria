import logging
import os
import pickle
import sys
import time
from typing import List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from sklearn.pipeline import Pipeline

from src.entities import HeartData, HeartResponse
from src.validate import is_data_valid

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

pipeline: Optional[Pipeline] = None

start_time = time.time()

app = FastAPI()


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_prediction(data: List[HeartData], pipeline: Pipeline) -> List[HeartResponse]:
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
    time.sleep(25)
    model_path = os.getenv("PATH_TO_MODEL", default="model.pkl")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)
    global pipeline
    pipeline = load_object(model_path)


@app.get("/status")
def status() -> bool:
    global start_time
    if time.time() - start_time > 120:
        raise RuntimeError
    return f"Pipeline is ready: {pipeline is not None}."


@app.api_route("/predict", response_model=List[HeartResponse], methods=["GET", "POST"])
def predict(request: List[HeartData]):
    for data in request:
        is_valid, error_message = is_data_valid(data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
    return make_prediction(request, pipeline)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
