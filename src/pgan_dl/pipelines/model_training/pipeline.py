"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import initialize, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            initialize,
            [
                "params:input_dir",
                "params:latent_size",
                "params:final_res",
                "params:negative_slope",
                "params:alpha_step",
                "params:batch_size",
                "params:lr",
                "params:num_workers",
            ],
            ['model', 'dataloader'],
            name='initialize'
        ),
        node(
            train_model,
            [
                "model",
                "dataloader",
                "params:max_epochs",
                "params:checkpoint_path",
                "params:loger_entity",
                "params:loger_name",
            ],
            ['trained_model', 'train_dataloader'],
            name='train_model'
        )
    ])
