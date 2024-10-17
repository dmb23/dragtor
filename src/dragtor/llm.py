from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from time import sleep
from typing import Self

from loguru import logger
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
        # Setting up session with retries
        self.session = self._setup_http_session()

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

    def _setup_http_session(self):
        # Set up retry strategy for health check and main request
        retry_strategy = Retry(
            total=5,  # Retry up to 5 times
            backoff_factor=5,  # Wait time increases with each retry (e.g., 5s, 10s)
            status_forcelist=[500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["GET", "POST"],  # Allow retries on GET and POST methods
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _server_health_check(self):
        url = f"http://{self._host}:{self._port}/health"

        # Health check
        try:
            server_health = self.session.get(url)
            logger.debug(f"Llama server health check: {server_health.status_code}")
            return server_health.status_code
        except requests.RequestException as e:
            logger.error(f"Failed to check server health: {e}")
            return None

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

        server_status = self._server_health_check()
        if server_status != 200:
            logger.error(f"Llama server is down with status code: {server_status}")
            return ""

        try:
            response = self.session.post(url, headers=headers, data=json.dumps(data))
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

        server_status = self._server_health_check()
        if server_status != 200:
            logger.error(f"Llama server is down with status code: {server_status}")
            return ""

        try:
            response = self.session.post(url, headers=headers, data=json.dumps(data))
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


@dataclass
class LocalDragtor:
    """Manage user requests by including context information and feeding them to LLMs."""

    # TODO: figure out an interface for managing cache files in the LlamaServerHandler

    llm: LlamaServerHandler = field(default_factory=LlamaServerHandler.from_config)
    user_prompt_template: str = config.conf.select("prompts.user_template")
    index: Index = field(default_factory=get_index)
    _questions: deque = field(init=False, default_factory=deque)
    _answers: deque = field(init=False, default_factory=deque)

    @property
    def system_prompt(self) -> str:
        return config.conf.select("prompts.system", "")

    def answer(self, question: str, **kwargs) -> str:
        """Generate an answer to the question, using the available knowledge.

        Use chunks from RAG retrieval as context."""
        context = self._get_context(question)
        prompt = self._expand_user_prompt(question, context)
        with self.llm:
            result = self.llm.query_llm(prompt, **kwargs)

        return result

    def _get_context(self, question: str, retrieve_chunks: bool = True) -> str:
        if retrieve_chunks:
            return "\n".join(self.index.query(question))

        return ""

    def _expand_user_prompt(self, question: str, context: str, is_chat: bool = False) -> str:
        """Infuse a question of the user with context"""
        prompt = self.user_prompt_template.format(context=context, question=question)
        logger.debug(f"built final prompt:\n{prompt}")
        if not is_chat and len(self.system_prompt) > 0:
            prompt = "\n\n".join([self.system_prompt, prompt])
        return prompt

    def _to_messages(self, question: str, context: str) -> list[dict]:
        first_prompt = self._expand_user_prompt(question, context, is_chat=True)
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": first_prompt,
            },
        ]
        return messages

    def chat(self, question: str, contextfile: str = "", **kwargs) -> str:
        """Generate an answer to the question via the chat interface.

        Use available knowledge via RAG approach.
        """
        if contextfile == "":
            context = self._get_context(question)
            messages = self._to_messages(question, context)
            with self.llm:
                result = self.llm.chat_llm(messages, **kwargs)
        else:
            context = Path(contextfile).resolve().read_text()
            context_id = hashlib.md5(context.encode("utf-8")).hexdigest()
            statefile = f"{context_id}.bin"
            messages = self._to_messages(question, context)
            result = self.llm.chat_from_state(messages, statefile, **kwargs)
        logger.debug("finished llm chat completion")

        return result
