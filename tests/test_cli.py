"""mostly to check that the commands run, and to allow to debug inside"""

import pytest

from dragtor.cli import Cli


@pytest.fixture
def question() -> str:
    return "How can I treat a mild A2 pulley strain?"


def test_cli_load(caplog, empty_store):
    Cli().load()
    print(caplog.text)
    assert "Loaded configured data successfully" in caplog.text
    pass


def test_cli_index(caplog, empty_store):
    Cli().index()
    assert "Indexed all cached data successfully" in caplog.text


def test_cli_search(question: str, full_store):
    found_texts = Cli().search(question)

    assert len(found_texts) > 0


# def test_cli_preload(full_store, caplog):
#     Cli().preload()
#
#     assert "Preloaded 1/" in caplog.text


def test_cli_ask(question: str, full_store):
    answer = Cli().ask(question)

    assert len(answer) > 0
