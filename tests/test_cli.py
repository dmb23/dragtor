"""mostly to check that the commands run, and to allow to debug inside"""

from dragtor.cli import Cli
import pytest


@pytest.fixture
def question() -> str:
    return "How can I treat a mild A2 pulley strain?"


@pytest.mark.skip(reason="TODO")
def test_cli_load(caplog):
    Cli().load()
    print(caplog.text)
    assert "Loaded data successfully" in caplog.text
    pass


@pytest.mark.skip(reason="TODO")
def test_cli_index(caplog):
    Cli().index()
    assert "Indexed all cached data successfully" in caplog.text


@pytest.mark.skip(reason="TODO")
def test_cli_search(question: str):
    found_texts = Cli().search(question)

    assert len(found_texts) > 0


@pytest.mark.skip(reason="TODO")
def test_cli_ask(question: str):
    answer = Cli().ask(question)

    assert len(answer) > 0
