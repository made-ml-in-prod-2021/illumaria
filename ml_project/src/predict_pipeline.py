import json
import logging
import os
import sys

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data import read_data, split_train_val_data
from src.entities.project_params import APPLICATION_NAME
from src.entities.predict_pipeline_params import (
    PredictPipelineParams,
    PredictPipelineParamsSchema,
)
from src.features.build_features import build_transformer, make_features
from src.models import (
    deserialize_model,
    predict_model,
)

logger = logging.getLogger(APPLICATION_NAME)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler("logs/predict.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def predict_pipeline(prediction_pipeline_params: PredictPipelineParams):
    """
    The pipeline to get model predictions on provided data.
    :param prediction_pipeline_params: prediction params
    :return: nothing
    """
    logger.info("Loading data...")
    data = pd.read_csv(prediction_pipeline_params.input_data_path)
    logger.info(f"Data shape is {data.shape}")

    logger.info("Loading model...")
    model = deserialize_model(prediction_pipeline_params.model_path)
    logger.info(f"Loaded model: {model}")

    feature_transformer = build_transformer(prediction_pipeline_params.feature_params)
    feature_transformer.fit(data)

    logger.info("Preparing features...")
    features, target = make_features(
        feature_transformer,
        data,
        prediction_pipeline_params.feature_params,
    )
    logger.info(f"Features shape is {features.shape}")

    logger.info("Making predictions...")
    predictions = predict_model(model, features)
    logger.info(f"Predictions shape is {predictions.shape}")

    data["predictions"] = predictions
    logger.info(f"Saving predictions to {prediction_pipeline_params.output_data_path}...")
    data.to_csv(prediction_pipeline_params.output_data_path)
    logger.info("Done.")


@hydra.main(config_path="../configs", config_name="predict_config.yaml")
def main(config: DictConfig):
    """
    Load prediction parameters from config file and start the prediction process.
    :return: nothing
    """
    os.chdir(hydra.utils.to_absolute_path('.'))
    schema = PredictPipelineParamsSchema()
    config = schema.load(config)
    logger.info(f"Prediction config:\n{OmegaConf.to_yaml(config)}")
    predict_pipeline(config)


if __name__ == "__main__":
    main()
