from pathlib import Path
from typing import Iterable

import requests

from dragtor.config import ConfigurationError, config

STATUS_OK = 200


class JinaLoader:
    _jina_base = "https://r.jina.ai/"

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

    def load_jina(self, urls: Iterable[str] | str) -> list[str]:
        """Load and cache text from Jina Reader API for a list of URLs"""
        full_texts = []
        outdir: Path = Path(config.base_path) / config.data.jina_cache
        if not outdir.is_dir():
            outdir.mkdir(parents=True)

        if isinstance(urls, str):
            urls = [urls]

        for url in urls:
            fpath = outdir / f"jina_{url.replace('/', '_')}.md"
            try:
                full_texts.append(fpath.read_text(encoding="utf8"))
            except IOError:
                full_text = self._load_jina_reader(url)
                fpath.write_text(full_text, encoding="utf8")
                full_texts.append(full_text)

        return full_texts
