"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            evaluate,
            ["trained_model", "train_dataloader", "params:generated_samples_no","params:batch_size"],
            ["fid"],
            name='evaluate_model'
        )
    ])
