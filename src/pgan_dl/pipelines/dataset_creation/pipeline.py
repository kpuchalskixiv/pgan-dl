"""
This is a boilerplate pipeline 'dataset_creation'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import get_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            get_data,
            ["params:raw_path"],
            None,
            name='get_data'
        )
    ])
