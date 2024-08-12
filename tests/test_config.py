from dragtor.config import config


def test_config():
    assert len(config.keys())
