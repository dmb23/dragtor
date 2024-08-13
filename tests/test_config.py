from dragtor.config import config


def test_config():
    """Just check that config is loaded and filled"""
    assert len(config.keys())
