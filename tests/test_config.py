from collections.abc import Iterable
from pathlib import Path

from dragtor import config
from dragtor.data import JinaLoader


def test_config_exists():
    """Just check that config is loaded and filled"""
    assert len(config.conf.keys())


def test_config_access():
    # check that basic properties are present
    assert isinstance(config.conf.base_path, str)
    assert isinstance(config.conf.data.blogs, Iterable)
    # check that we are working with test config
    assert "test blog" in config.conf.data.blogs
    assert len(config.conf.data.blogs) == 1
    # check that select method works as expected
    assert isinstance(config.conf.select("model.file_path"), str)
    assert config.conf.select("not.existing.variable", None) is None
    assert config.conf.select("not.existing.variable") is None


def test_config_mocking():
    """Check that the test config is used correctly in different locations"""
    assert "assets" in config.conf.base_path
    loader = JinaLoader()
    assert loader._cache_dir == (Path(config.conf.base_path) / config.conf.data.cache_dir)
