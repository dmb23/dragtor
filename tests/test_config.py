from collections.abc import Iterable

from dragtor.config import config


def test_config_exists():
    """Just check that config is loaded and filled"""
    assert len(config.keys())


def test_config_access():
    assert isinstance(config.base_path, str)
    assert isinstance(config.data.hoopers_urls, Iterable)
    assert isinstance(config.select("model.file_path"), str)
    assert config.select("not.existing.variable", None) is None
    assert config.select("not.existing.variable") is None
