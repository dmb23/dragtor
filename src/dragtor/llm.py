from dataclasses import dataclass, field
import json
from pathlib import Path
import shlex
import subprocess
from time import sleep
from typing import Self

from loguru import logger
import requests

from dragtor.config import config
from dragtor.index.index import Index, get_index

_default_system_prompt = """You are an assistant who provides advice on health care and training questions for climbers.
Please answer questions in 3 paragraphs."""

_default_user_prompt_template = """Please use the following pieces of context to answer the question.
context:
{context}

question:
{question}

answer:
"""


@dataclass
class LlamaHandler:
    modelpath: Path
    host: str = field(init=False)
    port: str = field(init=False)

    def __post_init__(self):
        self.host = config._select("model.host", default="127.0.0.1")
        port = config._select("model.port", default="8080")
        if type(port) is not str:
            port = f"{port:04d}"
        self.port = port
        self.modelpath = Path(self.modelpath)

    def _build_server_command(self) -> str:
        kwargs = config._select("model.kwargs", default={})
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

    def query(self, prompt: str, **kwargs) -> str:
        url = f"http://{self.host}:{self.port}/completion"
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": prompt,
            "n_predict": config._select("model.max_completion_tokens", default=128),
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

    @classmethod
    def from_config(cls) -> Self:
        modelpath = config._select("model.file_path", default=None)
        return cls(modelpath)


@dataclass
class Generator:
    # llm: Llama = field(default_factory=_model_loader)
    llm: LlamaHandler = field(default_factory=LlamaHandler.from_config)
    user_prompt_template: str = _default_user_prompt_template
    index: Index = field(default_factory=get_index)

    @property
    def system_prompt(self) -> str:
        return _default_system_prompt

    def query(self, question: str) -> str:
        """Generate an answer to the question, using the available knowledge."""
        # return self._create_chat_completion(question)
        return self._create_completion(question)

    def _expand_user_prompt(self, question: str) -> str:
        """Infuse a question of the user with context"""
        context = "\n".join(self.index.query(question))
        prompt = self.user_prompt_template.format(context=context, question=question)
        logger.debug(f"built final prompt:\n{prompt}")
        return prompt

    def _create_completion(self, question: str) -> str:
        prompt = self._expand_user_prompt(question)
        with LlamaHandler.from_config() as llm:
            result = llm.query(prompt)

        return result["choices"][0]["text"]

    # def _create_chat_completion(self, question: str) -> str:
    #     prompt = self._expand_user_prompt(question)
    #     max_tokens = config._select("model.max_completion_tokens", default=None)
    #     logger.debug(f"prompting model for chat completion for {max_tokens} tokens")
    # result = self.llm.create_chat_completion(
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": self.system_prompt,
    #         },
    #         {
    #             "role": "user",
    #             "content": prompt,
    #         },
    #     ],
    #     # max_tokens=max_tokens,
    # )
    # logger.debug("finished llm chat completion")
    #
    # return result["choices"][0]["message"]["content"]
