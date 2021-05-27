import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data import read_data
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
    logger.info("Loading data...")
    data = read_data(prediction_pipeline_params.input_data_path)

    logger.info("Loading pipeline...")
    pipeline = deserialize_model(prediction_pipeline_params.model_path)

    logger.info("Making predictions...")
    predictions = predict_model(pipeline, data)

    logger.info("Saving predictions...")
    data["predictions"] = predictions
    data.to_csv(prediction_pipeline_params.output_data_path)
    logger.info(f"Predictions saved to {prediction_pipeline_params.output_data_path}")
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
