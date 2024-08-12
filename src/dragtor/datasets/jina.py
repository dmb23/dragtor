from pathlib import Path
from typing import Any

import requests
from kedro.io import AbstractDataset

_STATUS_OK = 200


class JinaReaderDataset(AbstractDataset):
    def __init__(self, urls: list[str], filepath: str, creds: dict[str, str]) -> None:
        self._urls = urls
        self._filepath = Path(filepath)
        self._key = creds["api_key"]

    def _read_jina(self, url: str) -> str:
        jina_url = f"https://r.jina.ai/{url}"
        headers = {"Authorization": f"Bearer {self._key}"}

        response: requests.Response = requests.get(jina_url, headers=headers)
        if response.status_code != _STATUS_OK:
            raise ValueError(response)

        return response.text

    def _load(self) -> dict[str, str]:
        entries = {}
        for url in self._urls:
            fname = self._filepath / f"{url}.md"
            if fname.is_file():
                entries[url] = fname.read_text()
            else:
                entry = self._read_jina(url)
                entries[url] = entry
                fname.write_text(entry)
        return entries

    def _save(self) -> None:
        raise NotImplementedError(
            "JinaReaderDataset can not save data. All requests are cached to file."
        )

    def _describe(self) -> dict[str, Any]:
        """Returns a dict that describes the attributes of the Dataset"""
        return {
            "urls": self._urls,
            "filepath": str(self._filepath),
        }
