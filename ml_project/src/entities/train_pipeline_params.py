import yaml

from dataclasses import dataclass
from marshmallow_dataclass import class_schema

from .feature_params import FeatureParams
from .split_params import SplitParams
from .train_params import TrainParams


@dataclass()
class TrainPipelineParams:
    """Dataclass for training pipeline configuration."""
    input_data_path: str
    output_model_path: str
    metric_path: str
    split_params: SplitParams
    feature_params: FeatureParams
    train_params: TrainParams


TrainPipelineParamsSchema = class_schema(TrainPipelineParams)


def read_train_pipeline_params(path: str) -> TrainPipelineParams:
    """
    Read configuration file for training pipeline.
    :param path: path to config file
    :return: config dataclass
    """
    with open(path, "r") as input_stream:
        schema = TrainPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
