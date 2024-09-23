from functools import partial
import inspect
import logging
from pathlib import Path
from typing import cast

from loguru import logger
from omegaconf import DictConfig, OmegaConf


class InterceptHandler(logging.Handler):
    """Example from loguru documentation to intercept standard logging"""

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


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
def _select_hack(config_obj, *args, **kwargs):
    match len(args):
        case 1:
            return OmegaConf.select(config_obj, key=args[0], **kwargs)
        case 2:
            return OmegaConf.select(config_obj, key=args[0], default=args[1], **kwargs)
        case _:
            return OmegaConf.select(config_obj, *args, **kwargs)


super(DictConfig, config).__setattr__("select", partial(_select_hack, config))

# allow to config if logs from libraries should be sent to the loguru handler
if config.select("expose_library_logs", default=False):
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
