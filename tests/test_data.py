from pathlib import Path

import pytest

from dragtor import config
from dragtor.data.jina import JinaLoader


@pytest.mark.skip(reason="uses API credits")
def test_run_jina_loading():
    url = config.conf.data.jina_urls[0]

    fpath: Path = (
        Path(config.conf.base_path)
        / config.conf.data.jina_cache
        / f"jina_{url.replace('/', '_')}.md"
    )
    if fpath.exists() and fpath.is_file():
        fpath.unlink(missing_ok=True)
    JinaLoader().load_to_cache()

    assert fpath.exists()
    assert fpath.is_file()
    assert len(fpath.read_text()) > 0


@pytest.mark.skip(reason="Failed test with horrible test design...")
def test_jina_cache():
    """bad test design, relies on some file being cached. If it fails, you know where to look..."""
    loader = JinaLoader()
    full_texts = loader.get_cache()
    assert len(full_texts) > 0
    assert len(full_texts[0]) > 0
