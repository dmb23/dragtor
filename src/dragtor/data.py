from pathlib import Path
from typing import Iterable

from loguru import logger
import requests

from dragtor.config import ConfigurationError, config

STATUS_OK = 200


class JinaLoader:
    """Load text content of websites via Jina Reader API.

    This saves the hassle of parsing content of websites manually.
    This costs API credits, the first million are free.

    Requires `creds.jina` as a key in configuration (credentials.yml) with an API key
    """

    _jina_base = "https://r.jina.ai/"
    outdir: Path = Path(config.base_path) / config.data.jina_cache

    def _load_jina_reader(self, url: str) -> str:
        try:
            api_key = config.creds.jina
        except AttributeError:
            raise ConfigurationError("Expect `creds.jina` in configuration")

        jina_url = f"{self._jina_base}{url}"
        headers = {"Authorization": f"Bearer {api_key}"}

        response = requests.get(jina_url, headers=headers)

        if response.status_code != STATUS_OK:
            raise IOError(response)

        return response.text

    def load_jina_to_cache(self, urls: Iterable[str] | str) -> None:
        """Load text from Jina Reader API for a list of URLs and cache to files"""
        if not self.outdir.is_dir():
            self.outdir.mkdir(parents=True)

        if isinstance(urls, str):
            urls = [urls]

        counter = 0
        for url in urls:
            fpath = self.outdir / f"jina_{url.replace('/', '_')}.md"
            try:
                fpath.read_text(encoding="utf8")
                logger.debug(f"Already cached {url}")
            except IOError:
                full_text = self._load_jina_reader(url)
                fpath.write_text(full_text, encoding="utf8")
                logger.debug(f"Loaded {url} from Jina Reader API")
                counter += 1

        logger.info(f"Loaded {counter} urls via Jina Reader API")

    def get_cache(self) -> list[str]:
        """Get all previously cached text that was loaded to file"""
        full_texts = []
        for fpath in self.outdir.glob("*.md"):
            full_texts.append(fpath.read_text(encoding="utf8"))

        return full_texts
