from pathlib import Path

from dragtor.config import config
from dragtor.data import JinaLoader
import pytest


@pytest.mark.skip(reason="uses API credits")
def test_full_jina_loading():
    url = config.data.hoopers_urls[0]

    fpath: Path = Path(config.base_path) / config.data.jina_cache / f"jina_{url}.md"
    if fpath.exists() and fpath.is_file():
        fpath.unlink(missing_ok=True)
    full_text = JinaLoader().load_jina(url)

    assert len(full_text)
