import json
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data import read_data, split_train_val_data
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
from src.utils import setup_logger

logger = setup_logger(path="logs/train.log")


def train_pipeline(training_pipeline_params: TrainPipelineParams):
    """
    The pipeline to train and evaluate model and store artifacts.
    :param training_pipeline_params: training params
    :return: nothing
    """
    logger.info(f"Start train pipeline with params {training_pipeline_params}.")

    data = read_data(training_pipeline_params.input_data_path)
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.split_params
    )

    logger.info("Building transformer...")
    transformer = build_transformer(training_pipeline_params.feature_params)
    logger.info("Fitting transformer...")
    transformer.fit(train_df)

    logger.info("Preparing train data...")
    train_features, train_target = make_features(
        transformer, train_df, training_pipeline_params.feature_params
    )

    logger.info("Training model...")
    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )

    logger.info("Preparing validation data...")
    val_features, val_target = make_features(
        transformer, val_df, training_pipeline_params.feature_params
    )

    logger.info("Making predictions on validation data...")
    predicts = predict_model(
        model,
        val_features,
    )

    logger.info("Evaluating model...")
    metrics = evaluate_model(
        predicts,
        val_target,
    )

    logger.info("Saving metrics...")
    with open(training_pipeline_params.metrics_path, "w") as metric_file:
        json.dump(metrics, metric_file)

    logger.info("Saving pipeline...")
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
