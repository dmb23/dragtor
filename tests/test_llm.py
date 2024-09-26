from dragtor import llm
import pytest


@pytest.fixture(scope="module")
def llama_server_handler():
    llama_handler = llm.LlamaServerHandler.from_config()
    with llama_handler:
        yield llama_handler

@pytest.fixture
def llama_cli_handler():
    return llm.LlamaCliHandler.from_config()

def test_gen_query_unrestricted(llama_server_handler):
    res = llama_server_handler.query_llm("What is the meaning of life?", n_predict=10, ignore_eos=True)

    assert type(res) is str
    assert len(res) > 0


def test_gen_query_eos(llama_server_handler):
    res = llama_server_handler.query_llm("What is the meaning of", n_predict=2048, temperature=0)

    assert type(res) is str
    assert len(res) == 6
    assert res == " life?"


def test_gen_query_unrestricted_cli(llama_cli_handler):
    res = llama_cli_handler.query_llm("What is the meaning of life?", n_predict=10, ignore_eos=True)

    assert type(res) is str
    assert len(res) > 0


def test_gen_query_eos_cli(llama_cli_handler):
    res = llama_cli_handler.query_llm("What is the meaning of", n_predict=2048, temperature=0)

    assert type(res) is str
    assert len(res) == 6
    assert res == " life?"


def test_dragtor_answer():
    # config.conf.model.max_completion_tokens = 64
    ld = llm.LocalDragtor()

    res = ld.answer("What is the meaning of life?")

    assert len(res) > 0


def test_dragtor_chat():
    ld = llm.LocalDragtor()

    res = ld.chat("What is the meaning of life?")

    assert len(res) > 0
