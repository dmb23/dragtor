from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import shlex
import subprocess
from time import sleep
from typing import Self

from loguru import logger
import requests

from dragtor import config
from dragtor.index.index import Index, get_index


@dataclass
class LlamaServerHandler:
    """Manage interaction with llama.cpp server

    A server is started in the shell via a subprocess.
    Then any interaction can just happen via http requests.

    The class is designed so that an instance can be used as a context manager:
    ```python
    llm = LlamaHandler(modelpath)
    with llm:
        response = llm.query(prompt)
    ```

    Requires llama.cpp installed and the executables to be findable in a shell.
    """

    modelpath: Path
    _host: str = field(init=False)
    _port: str = field(init=False)
    _state_dir: Path = field(init=False)

    def __post_init__(self):
        self._host = config.conf.select("model.host", default="127.0.0.1")
        port = config.conf.select("model.port", default="8080")
        if type(port) is not str:
            port = f"{port:04d}"
        self._port = port
        self.url = f"http://{self._host}:{self._port}"
        self.modelpath = Path(self.modelpath)
        self._state_dir = Path(
            config.conf.select("model.state_dir", default=f"{config.conf.base_path}/checkpoints")
        )

    def _build_server_command(self) -> str:
        """Build the shell command to start the llama.cpp server"""
        kwargs = config.conf.select("model.kwargs", default={})
        pieces = [
            "llama-server",
            "-m",
            str(self.modelpath.resolve()),
            "--host",
            self._host,
            "--port",
            self._port,
            "--slot-save-path",
            self._state_dir,
        ]

        if len(kwargs):
            for k, v in kwargs.items():
                pieces.extend([f"--{k}", str(v)])
        return shlex.join(pieces)

    def __enter__(self):
        _cmd = self._build_server_command()
        logger.debug(f"starting Llama server with command {_cmd}")
        self.p = subprocess.Popen(_cmd, shell=True)
        sleep(0.5)

    def __exit__(self, exc_type, exc_value, traceback):
        self.p.terminate()

    def query_llm(self, prompt: str, **kwargs) -> str:
        """Send a query to the llama server.

        Will fail if the server is not started,
        i.e. not run inside a context created by the class.
        """
        url = f"{self.url}/completion"
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": prompt,
            "n_predict": config.conf.select("model.max_completion_tokens", default=128),
        }
        data.update(kwargs)

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            if response.status_code != 200:
                logger.error(f"Recieved response status {response.status_code}")
            result = response.json()
            return result["content"]
        except requests.RequestException as e:
            logger.error(f"Error querying Llama server: {e}")
            return ""

    def chat_llm(self, messages: list[dict], **kwargs) -> str:
        url = f"{self.url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "messages": messages,
            "n_predict": config.conf.select("model.max_completion_tokens", default=128),
        }
        data.update(kwargs)

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            if response.status_code != 200:
                logger.error(f"Recieved response status {response.status_code}")
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            logger.error(f"Error querying Llama server: {e}")
            return ""

    def _save_state(self, statefile: Path, slot_id: int = 0):
        """POST to  /slots/{slot_id}?action=save"""
        url = f"{self.url}/slots/{slot_id}?action=save"
        headers = {"Content-Type": "application/json"}
        data = {
            "filename": statefile.resolve(),
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            if response.status_code != 200:
                logger.error(f"Recieved response status {response.status_code}")
            logger.info(response)
        except requests.RequestException as e:
            logger.error(f"Error querying Llama server: {e}")

    def _load_state(self, statefile: Path, slot_id: int = 0):
        """Load a saved state from the llama server"""
        url = f"{self.url}/slots/{slot_id}?action=restore"
        headers = {"Content-Type": "application/json"}
        data = {
            "filename": statefile.resolve(),
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            if response.status_code != 200:
                logger.error(f"Recieved response status {response.status_code}")
            logger.info(response)
        except requests.RequestException as e:
            logger.error(f"Error querying Llama server: {e}")

    @classmethod
    def from_config(cls) -> Self:
        modelpath = config.conf.select("model.file_path", default=None)
        return cls(modelpath)


@dataclass
class LlamaCliHandler:
    """Manage interaction with llama.cpp using the llama-cli command directly

    Requires llama.cpp installed and the executables to be findable in the system PATH.
    """

    modelpath: Path
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.modelpath = Path(self.modelpath)

    def _build_cli_command(self, prompt: str, **kwargs) -> str:
        """Build the shell command to run llama-cli"""
        pieces = [
            "llama-cli",
            "-m",
            str(self.modelpath.resolve()),
            "-p",
            prompt,
        ]

        cli_kwargs = self.kwargs.copy()
        cli_kwargs.update(kwargs)

        for k, v in cli_kwargs.items():
            if type(v) is bool and v:
                pieces.extend([f"--{k}"])
            else:
                pieces.extend([f"--{k}", str(v)])

        return shlex.join(pieces)

    def query_llm(self, prompt: str, **kwargs) -> str:
        """Send a query to llama-cli."""
        cmd = self._build_cli_command(prompt, **kwargs)

        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running llama-cli: {e}")
            return ""

    @classmethod
    def from_config(cls) -> "LlamaCliHandler":
        modelpath = config.conf.select("model.file_path", default=None)
        kwargs = config.conf.select("model.kwargs", default={})
        return cls(modelpath, kwargs)


@dataclass
class LocalDragtor:
    """Manage user requests by including context information and feeding them to LLMs."""

    llm: LlamaServerHandler = field(default_factory=LlamaServerHandler.from_config)
    user_prompt_template: str = config.conf.select("prompts.user_template")
    index: Index = field(default_factory=get_index)
    _questions: deque = field(init=False, default_factory=deque)
    _answers: deque = field(init=False, default_factory=deque)

    @property
    def system_prompt(self) -> str:
        return config.conf.select("prompts.system", "")

    def answer(self, question: str, **kwargs) -> str:
        """Generate an answer to the question, using the available knowledge."""
        prompt = self._expand_user_prompt(question)
        with self.llm:
            result = self.llm.query_llm(prompt, **kwargs)

        return result

    def _expand_user_prompt(self, question: str, is_chat: bool = False) -> str:
        """Infuse a question of the user with context"""
        context = "\n".join(self.index.query(question))
        prompt = self.user_prompt_template.format(context=context, question=question)
        logger.debug(f"built final prompt:\n{prompt}")
        if not is_chat and len(self.system_prompt) > 0:
            prompt = "\n\n".join([self.system_prompt, prompt])
        return prompt

        #     {
        #         "role": "system",
        #         "content": self.system_prompt,
        #     },
        #     {
        #         "role": "user",
        #         "content": first_prompt,
        #     },
        # ]
        # # TODO: append older question / answer pairs if any interface for chat exists
        # with self.llm:
        #     result = self.llm.chat_llm(messages, **kwargs)
        # logger.debug("finished llm chat completion")
        #
        # return result
