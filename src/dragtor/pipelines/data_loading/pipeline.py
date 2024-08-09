from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_jina_reader


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_jina_reader,
                inputs="params:addr",
                outputs="blog_post",
                name="load_jina_example",
            )
        ]
    )
