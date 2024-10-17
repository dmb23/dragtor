import re

import pytest

from dragtor import llm


@pytest.fixture(scope="module")
def llama_server_handler():
    llama_handler = llm.LlamaServerHandler.from_config()
    with llama_handler:
        yield llama_handler


def test_gen_query_unrestricted(llama_server_handler):
    res = llama_server_handler.query_llm(
        "What is the meaning of life?", n_predict=10, ignore_eos=True
    )

    assert type(res) is str
    assert len(res) > 0


def test_gen_query_eos(llama_server_handler):
    res = llama_server_handler.query_llm("What is the meaning of", n_predict=2048, temperature=0)

    assert type(res) is str
    assert len(res) == 6
    assert res == " life?"


def test_store_management(capfd, tmp_path):
    lsh = llm.LlamaServerHandler.from_config()
    lsh._checkpoint_dir = tmp_path
    cache_file = tmp_path / "test.bin"
    messages = [
        {"role": "system", "content": "You complete the last words of questions"},
        {"role": "user", "content": "What is the meaning of"},
    ]

    with lsh:
        lsh.chat_llm(messages, temperature=0)
    captured = capfd.readouterr()
    pattern = re.compile(r"prompt eval time[^\n]* (\d+) tokens")
    match = pattern.search(captured.out)
    assert match
    assert match.group(1) == "26"

    with capfd.disabled():
        lsh.store_state(messages, cache_file.name)

    lsh.chat_from_state(messages, cache_file.name)

    captured2 = capfd.readouterr()
    match2 = pattern.search(captured2.out)
    assert match2
    assert match2.group(1) == "1"


def test_dragtor_answer():
    # config.conf.model.max_completion_tokens = 64
    ld = llm.LocalDragtor()

    res = ld.answer("What is the meaning of life?")

    assert len(res) > 0


def test_dragtor_chat():
    ld = llm.LocalDragtor()

    res = ld.chat("What is the meaning of life?")

    assert len(res) > 0
