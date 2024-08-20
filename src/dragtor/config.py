from functools import partial
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf


class ConfigurationError(Exception):
    pass


_conf_root = Path(__file__).parent.parent.parent / "config"
if not _conf_root.is_dir():
    raise ConfigurationError("Could not find configuration directory at %s", _conf_root)

_valid_confs = []
for conf_file in _conf_root.glob("**/*"):
    try:
        _valid_confs.append(OmegaConf.load(conf_file))
    except (TypeError, IOError):
        pass

config = OmegaConf.merge(*_valid_confs)
config = cast(DictConfig, config)

# hooray for Python hacks!
super(DictConfig, config).__setattr__("_select", partial(OmegaConf.select, config))
