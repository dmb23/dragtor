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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dragtor import config
from dragtor.index.index import Index, get_index


@dataclass
class LlamaHandler:
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
    host: str = field(init=False)
    port: str = field(init=False)

    def __post_init__(self):
        self.host = config.conf.select("model.host", default="127.0.0.1")
        port = config.conf.select("model.port", default="8080")
        if type(port) is not str:
            port = f"{port:04d}"
        self.port = port
        self.modelpath = Path(self.modelpath)
        # Setting up session with retries
        self.session = self._setup_http_session()

    def _build_server_command(self) -> str:
        """Build the shell command to start the llama.cpp server"""
        kwargs = config.conf.select("model.kwargs", default={})
        pieces = [
            "llama-server",
            "-m",
            str(self.modelpath.resolve()),
            "--host",
            self.host,
            "--port",
            self.port,
        ]

        if len(kwargs):
            for k, v in kwargs.items():
                pieces.extend([k, str(v)])
        return shlex.join(pieces)

    def __enter__(self):
        _cmd = self._build_server_command()
        logger.debug(f"starting Llama server with command {_cmd}")
        self.p = subprocess.Popen(_cmd, shell=True)
        sleep(0.5)

    def __exit__(self, exc_type, exc_value, traceback):
        self.p.terminate()

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
        url = f"http://{self.host}:{self.port}/health"

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
        url = f"http://{self.host}:{self.port}/completion"
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
        url = f"http://{self.host}:{self.port}/chat/completions"
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
                logger.error(f"Received response status {response.status_code}")
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            logger.error(f"Error querying Llama server: {e}")
            return ""

    @classmethod
    def from_config(cls) -> Self:
        modelpath = config.conf.select("model.file_path", default=None)
        return cls(modelpath)


@dataclass
class LocalDragtor:
    """Manage user requests by including context information and feeding them to LLMs."""

    llm: LlamaHandler = field(default_factory=LlamaHandler.from_config)
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

    def chat(self, question: str, **kwargs) -> str:
        """Use the chat interface to answer a question of the user with context"""
        self._questions.append(question)
        first_question = self._questions.popleft()
        first_prompt = self._expand_user_prompt(first_question, is_chat=True)
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
        # TODO: append older question / answer pairs if any interface for chat exists
        with self.llm:
            result = self.llm.chat_llm(messages, **kwargs)
        logger.debug("finished llm chat completion")

        return result
