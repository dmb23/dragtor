from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger

from dragtor import config


class DataLoader(ABC):
    def __init__(self, cache_glob: str = "*.txt", cache_dir: Path | None = None):
        self._cache_dir: Path
        self._cache_glob: str
        if cache_dir is not None:
            self._cache_dir = cache_dir
        else:
            self._cache_dir = Path(config.conf.base_path) / config.conf.data.cache_dir
        self._cache_dir.mkdir(exist_ok=True, parents=True)
        self._cache_glob = cache_glob

    @abstractmethod
    def load_to_cache(self) -> None:
        """Takes sources defined in config and loads them to a cache of text files"""
        pass

    def get_cache(self) -> list[str]:
        """Get all previously cached text that was loaded to file"""
        full_texts = []
        for fpath in self._cache_dir.glob(self._cache_glob):
            logger.debug(f"loading cached file {fpath}")
            full_texts.append(fpath.read_text(encoding="utf8"))

        return full_texts
