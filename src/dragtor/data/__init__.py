# ruff: noqa: I001
from dragtor.data.data import DataLoader
from dragtor.data.audio import AudioLoader
from dragtor.data.jina import JinaLoader


def get_all_loaders() -> list[DataLoader]:
    return [
        JinaLoader(),
        AudioLoader(),
    ]
