from importlib.resources import as_file, files

from _pytest.logging import LogCaptureFixture
from dragtor import config
from dragtor.index import store
from loguru import logger
from omegaconf import OmegaConf
import pytest


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Send loguru logs to caplog"""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture(autouse=True, scope="session")
def mock_config():
    source = files("tests.assets").joinpath("test_params.yml")

    with as_file(source) as test_conf_path:
        test_config = OmegaConf.load(test_conf_path)

    config.conf.update(test_config)


@pytest.fixture
def empty_store() -> store.ChromaDBStore:
    vstore = store.get_store()
    if vstore.collection.count() > 0:
        vstore.collection.delete(vstore.collection.get()["ids"])

    return vstore
