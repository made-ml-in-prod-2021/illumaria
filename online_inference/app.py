import logging
import os
import pickle
import sys
from typing import List, Union, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conlist
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


class HeartModel(BaseModel):
    data: List[conlist(Union[int, float, str, None])]
    features: List[str]


class HeartResponse(BaseModel):
    id: int
    target: int


pipeline: Optional[Pipeline] = None


def make_prediction(
    data: List,
    features: List[str],
    pipeline: Pipeline,
) -> List[HeartResponse]:
    data = pd.DataFrame(data, columns=features)
    ids = [int(x) for x in data.index]
    predicts = pipeline.predict(data)

    return [
        HeartResponse(id=id_, target=int(target_))
        for id_, target_ in zip(ids, predicts)
    ]


@app.get("/")
def main():
    return "This is the entry point of our predictor."


@app.on_event("startup")
def load_model():
    model_path = os.getenv("PATH_TO_MODEL")
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
def predict(request: HeartModel):
    return make_prediction(request.data, request.features, pipeline)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
