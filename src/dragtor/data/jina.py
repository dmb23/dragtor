from loguru import logger
import requests

from dragtor import config
from dragtor.data import DataLoader

STATUS_OK = 200


class JinaLoader(DataLoader):
    """Load text content of websites via Jina Reader API.

    This saves the hassle of parsing content of websites manually.
    This costs API credits, the first million are free.

    Requires `creds.jina` as a key in configuration (credentials.yml) with an API key
    """

    def __init__(self, *args, **kwargs):
        kwargs.update({"cache_glob": "jina_*.md"})
        super().__init__(*args, **kwargs)
        self._jina_base: str = "https://r.jina.ai/"

    def _load_jina_reader(self, url: str) -> str:
        try:
            api_key = config.conf.creds.jina
        except AttributeError:
            raise config.ConfigurationError("Expect `creds.jina` in configuration")

        jina_url = f"{self._jina_base}{url}"
        headers = {"Authorization": f"Bearer {api_key}"}

        response = requests.get(jina_url, headers=headers)

        if response.status_code != STATUS_OK:
            raise IOError(response)

        return response.text

    def load_to_cache(self) -> None:
        """Load text from Jina Reader API for a list of URLs and cache to files"""
        urls = config.conf.data.jina_urls
        if isinstance(urls, str):
            urls = [urls]
        logger.debug(f"loading jina reader urls:\n{urls}")

        counter = 0
        for url in urls:
            fpath = self._cache_dir / f"jina_{url.replace('/', '_')}.md"
            try:
                fpath.read_text(encoding="utf8")
                logger.debug(f"Already cached {url}")
            except IOError:
                full_text = self._load_jina_reader(url)
                fpath.write_text(full_text, encoding="utf8")
                logger.debug(f"Loaded {url} from Jina Reader API")
                counter += 1

        logger.info(f"Loaded {counter} urls via Jina Reader API")
