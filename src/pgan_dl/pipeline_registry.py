"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from .pipelines import dataset_creation as dc
from .pipelines import model_training as mt
from .pipelines import model_evaluation as me


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    dataset_creation_pipeline = dc.create_pipeline()
    model_training_pipeline = mt.create_pipeline()
    model_evaluation_pipeline = me.create_pipeline()

    return {
        "dataset_creation": dataset_creation_pipeline,
        "model_training": model_training_pipeline,
        "model_evaluation": model_evaluation_pipeline,
        "__default__": dataset_creation_pipeline + model_training_pipeline + model_evaluation_pipeline,
    }