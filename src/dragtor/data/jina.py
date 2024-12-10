from loguru import logger
import requests

from dragtor import config
from dragtor.data.data import Document, DataLoader


class JinaLoader(DataLoader):
    """Load text content of websites via Jina Reader API.

    This saves the hassle of parsing content of websites manually.
    This costs API credits, the first million are free.
    TODO: switch to local model for website parsing

    Requires `creds.jina` as a key in configuration (credentials.yml) with an API key
    """

    def __init__(self, *args, **kwargs):
        kwargs.update({"cache_glob": "jina_*.json"})  # Change to JSON to store metadata
        super().__init__(*args, **kwargs)
        self._jina_base: str = "https://r.jina.ai/"

    def _load_jina_reader(self, url: str, metadata: dict) -> Document:
        try:
            api_key = config.conf.creds.jina
        except AttributeError:
            raise config.ConfigurationError("Expect `creds.jina` in configuration")

        jina_url = f"{self._jina_base}{url}"
        headers = {"Authorization": f"Bearer {api_key}"}

        response = requests.get(jina_url, headers=headers)
        response.raise_for_status()

        return Document(
            content=response.text,
            title=metadata.get("title", url.split("/")[-1]),
            id=f"jina_{url.replace('/', '_')}",
            author=metadata.get("author"),
            metadata={
                "source": url,
                "source_type": "blog",
                **metadata
            }
        )

    def load_to_cache(self) -> None:
        """Load text and metadata from Jina Reader API for URLs and cache to files"""
        blog_config = config.conf.data.blogs
        
        counter = 0
        for blog_name, blog_info in blog_config.items():
            author = blog_info.get("author")
            for url in blog_info.get("entries", []):
                metadata = {
                    "blog_name": blog_name,
                    "author": author,
                }
                
                fpath = self._cache_dir / f"jina_{url.replace('/', '_')}.json"
                try:
                    # Skip if already cached
                    fpath.read_text(encoding="utf8")
                    logger.debug(f"Already cached {url}")
                except IOError:
                    document = self._load_jina_reader(url, metadata)
                    # Store as JSON to preserve metadata
                    fpath.write_text(
                        document.json(ensure_ascii=False), 
                        encoding="utf8"
                    )
                    logger.debug(f"Loaded {url} from Jina Reader API")
                    counter += 1

        logger.info(f"Loaded {counter} urls via Jina Reader API")
    def get_cache(self) -> list[Document]:
        """Override parent method to handle JSON documents with metadata"""
        documents = []
        for fpath in self._cache_dir.glob(self._cache_glob):
            logger.debug(f"loading cached file {fpath}")
            # Parse JSON back into Document
            document = Document.parse_file(fpath)
            documents.append(document)

        return documents
