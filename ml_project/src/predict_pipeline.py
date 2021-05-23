import os

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.entities.predict_pipeline_params import (
    PredictPipelineParams,
    PredictPipelineParamsSchema,
)
from src.models import (
    deserialize_model,
    predict_model,
)
from src.utils import setup_logger

logger = setup_logger(path="logs/predict.log")


def predict_pipeline(prediction_pipeline_params: PredictPipelineParams):
    """
    The pipeline to get model predictions on provided data.
    :param prediction_pipeline_params: prediction params
    :return: nothing
    """
    data = pd.read_csv(prediction_pipeline_params.input_data_path)
    logger.info(f"Data shape is {data.shape}")

    pipeline = deserialize_model(prediction_pipeline_params.model_path)
    logger.info(f"Loaded pipeline: {pipeline}")

    predictions = predict_model(pipeline, data)
    logger.info(f"Predictions shape is {predictions.shape}")

    data["predictions"] = predictions
    data.to_csv(prediction_pipeline_params.output_data_path)
    logger.info(f"Saved predictions to {prediction_pipeline_params.output_data_path}")


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
