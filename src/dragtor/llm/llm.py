from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import shlex
import subprocess
from time import sleep
from typing import Self

from loguru import logger
import requests

from dragtor import config


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
    url: str = field(init=False)
    _checkpoint_dir: Path = field(init=False)
    _kwargs: dict = field(init=False)
    _temp_kwargs: dict = field(init=False, default_factory=dict)

    class SlotAction(Enum):
        SAVE = "save"
        RESTORE = "restore"
        ERASE = "erase"

    def __post_init__(self):
        self._host = config.conf.select("model.host", default="127.0.0.1")
        port = config.conf.select("model.port", default="8080")
        if type(port) is not str:
            port = f"{port:04d}"
        self._port = port
        self.url = f"http://{self._host}:{self._port}"
        self.modelpath = Path(self.modelpath)

        self._checkpoint_dir = Path(config.conf.base_path) / "checkpoints"
        if not self._checkpoint_dir.exists():
            self._checkpoint_dir.mkdir(parents=True)

        self._kwargs = config.conf.select("model.kwargs", default={})
        self._kwargs.update(
            {
                "model": str(self.modelpath.resolve()),
                "host": self._host,
                "port": self._port,
                "slot_save_path": str(self._checkpoint_dir.resolve()),
            }
        )

    def __call__(self, **kwargs):
        self._temp_kwargs.update({k.replace("_", "-"): v for k, v in kwargs})
        return self

    def _build_server_command(self) -> str:
        """Build the shell command to start the llama.cpp server"""
        self._kwargs.update(self._temp_kwargs)

        pieces = [
            "llama-server",
        ]

        for k, v in self._kwargs.items():
            if type(v) is bool:
                if v:
                    pieces.extend([f"--{k}"])
                else:
                    pieces.extend([f"--no-{k}"])
            else:
                pieces.extend([f"--{k}", str(v)])

        return shlex.join(pieces)

    def __enter__(self):
        _cmd = self._build_server_command()
        logger.debug(f"starting Llama server with command {_cmd}")
        self.p = subprocess.Popen(_cmd, shell=True)
        sleep(0.5)

    def __exit__(self, exc_type, exc_value, traceback):
        self.p.terminate()
        self._temp_kwargs.clear()

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

    def _manage_slot_state(self, action: SlotAction, filename: str = "", slot_id: int = 0):
        if action == LlamaServerHandler.SlotAction.SAVE and filename == "":
            logger.error("You need to provide a file to store a model state to.")
            return
        if action == LlamaServerHandler.SlotAction.RESTORE and filename == "":
            logger.error("You need to provide a file to restore a model state from.")
            return
        if action == LlamaServerHandler.SlotAction.ERASE and filename != "":
            logger.warning(
                "You provided a file for model state, ignored for erasing the slot memory!"
            )

        url = f"{self.url}/slots/{slot_id}?action={action.value}"
        kwargs = {}
        kwargs["headers"] = {"Content-Type": "application/json"}
        if action != LlamaServerHandler.SlotAction.ERASE:
            kwargs["data"] = json.dumps(
                {
                    "filename": filename,
                }
            )

        try:
            logger.debug(f"{url=}")
            logger.debug(f"{kwargs=}")
            response = requests.post(url, **kwargs)
            response.raise_for_status()
            if response.status_code != 200:
                logger.error(
                    f"Recieved response status {response.status_code} when managing slot state"
                )
        except requests.RequestException as e:
            logger.error(f"Error querying Llama server: {e}")

    def store_state(self, messages: list[dict], filename: str):
        statefile = Path(self._checkpoint_dir) / filename
        statefile.parent.mkdir(parents=True, exist_ok=True)
        statefile.unlink(missing_ok=True)
        statefile.touch()
        with self:
            self.chat_llm(messages, cache_prompt=True)
            self._manage_slot_state(
                LlamaServerHandler.SlotAction.SAVE, slot_id=0, filename=filename
            )

    def chat_from_state(self, messages: list[dict], filename: str) -> str:
        with self:
            self._manage_slot_state(
                LlamaServerHandler.SlotAction.RESTORE, slot_id=0, filename=filename
            )
            response = self.chat_llm(messages, cache_prompt=True)

        return response

    @classmethod
    def from_config(cls) -> Self:
        modelpath = config.conf.select("model.file_path", default=None)
        return cls(modelpath)
