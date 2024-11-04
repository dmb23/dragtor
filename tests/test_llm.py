import re
import subprocess
import pytest
import shutil

from pathlib import Path
from dragtor import config
from dragtor import llm
from dragtor.utils import Messages


def test_llama_server_availability():
    """Test if llama-server is executable."""
    exe_file = Path(config.conf.executables.llama_project) / "llama-server"

    # Check if the file exists
    if not exe_file.is_file():
        pytest.fail(
            f"llama-server executable not found at {exe_file}. Ensure the llama.cpp project path is configured correctly.")

    # Check if the file is executable
    if not shutil.which(str(exe_file)):
        pytest.fail(
            f"llama-server executable at {exe_file} is not recognized as executable. Verify permissions and path configuration.")

    try:
        result = subprocess.run([exe_file, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode == 0
    except FileNotFoundError as e:
        pytest.fail(f"Failed to run llama-server executable at {exe_file}: {e}")


@pytest.fixture(scope="class")
def llama_server_handler():
    llama_handler = llm.LlamaServerHandler.from_config()
    with llama_handler:
        yield llama_handler


class TestQueriesWithPersistentServer:
    def test_gen_query_unrestricted(self, llama_server_handler):
        res = llama_server_handler.query_llm(
            "What is the meaning of life?", n_predict=10, ignore_eos=True
        )

        assert type(res) is str
        assert len(res) > 0

    def test_gen_query_eos(self, llama_server_handler):
        res = llama_server_handler.query_llm(
            "What is the meaning of", n_predict=2048, temperature=0
        )

        assert type(res) is str
        assert len(res) == 6
        assert res == " life?"


def test_store_management(capfd, tmp_path):
    lsh = llm.LlamaServerHandler.from_config()
    lsh._checkpoint_dir = tmp_path
    # make shure the test is independent of a potentailly existing other server
    lsh._port = "51315"
    cache_file = tmp_path / "test.bin"
    messages = Messages()
    messages.system("You complete the last words of questions")
    messages.user("What is the meaning of")

    with lsh:
        lsh.chat_llm(messages, temperature=0)
    captured = capfd.readouterr()
    pattern = re.compile(r"prompt eval time[^\n]* (\d+) tokens")
    match = pattern.search(captured.out)
    assert match
    assert match.group(1) == "26"

    with capfd.disabled():
        with lsh:
            lsh.store_state(messages, cache_file.name)

    with lsh:
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
