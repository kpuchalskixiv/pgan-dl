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
            ["params:model_path", "params:input_dir", "params:generated_samples_no","params:batch_size", "params:num_workers"],
            ["fid"],
            name='evaluate_model'
        )
    ])
