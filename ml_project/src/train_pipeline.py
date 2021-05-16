import json
import logging
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data import read_data, split_train_val_data
from src.entities.project_params import APPLICATION_NAME
from src.entities.train_pipeline_params import (
    TrainPipelineParams,
    TrainPipelineParamsSchema,
)
from src.features.build_features import build_transformer, make_features
from src.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)

logger = logging.getLogger(APPLICATION_NAME)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler("logs/train.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def train_pipeline(training_pipeline_params: TrainPipelineParams):
    """
    The pipeline to train and evaluate model and store artifacts.
    :param training_pipeline_params: training params
    :return: nothing
    """
    logger.info(f"Start train pipeline with params {training_pipeline_params}.")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.split_params
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    train_features, train_target = make_features(
        transformer, train_df, training_pipeline_params.feature_params
    )

    logger.info(f"train_features.shape is {train_features.shape}")

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    val_features, val_target = make_features(
        transformer, val_df, training_pipeline_params.feature_params
    )

    logger.info(f"val_features.shape is {val_features.shape}")
    predicts = predict_model(
        model,
        val_features,
    )

    metrics = evaluate_model(
        predicts,
        val_target,
    )

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics are: {metrics}")

    path_to_model = serialize_model(model, training_pipeline_params.output_model_path, transformer)

    logger.info("Done.")

    return path_to_model, metrics


@hydra.main(config_path="../configs", config_name="train_config.yaml")
def main(config: DictConfig):
    """
    Load training parameters from config file and start training process.
    :return: nothing
    """
    os.chdir(hydra.utils.to_absolute_path('.'))
    schema = TrainPipelineParamsSchema()
    config = schema.load(config)
    logger.info(f"Training config:\n{OmegaConf.to_yaml(config)}")
    train_pipeline(config)


if __name__ == "__main__":
    main()
