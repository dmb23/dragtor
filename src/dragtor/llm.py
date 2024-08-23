from dataclasses import dataclass, field
from pathlib import Path
import shlex
import subprocess

from loguru import logger

from dragtor.config import config
from dragtor.index.index import Index, get_index

_default_system_prompt = "You are an assistant who provides advice on health care and training questions for climbers. Please answer questions in 3 paragraphs."

_default_user_prompt_template = """Please use the following pieces of context to answer the question.
context:
{context}

question:
{question}

answer:
"""


def _model_loader():
    logger.debug("starting initialization of model")
    llm = Llama(config.model.file_path, **config.model.kwargs)
    logger.debug("initialized model")
    return llm


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

    def _build_server_command(self) -> str:
        kwargs = config._select("model.kwargs", {})
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
                pieces.extend([k, v])
        return shlex.join(pieces)

    def __enter__(self):
        _cmd = self._build_server_command()
        logger.debug(f"starting Llama server with command {_cmd}")
        self.p = subprocess.Popen(_cmd, shell=True)

    def __exit__(self, exc_type, exc_value, traceback):
        self.p.terminate()

    def _query_model(prompt: str) -> str:
        pass


@dataclass
class Generator:
    # llm: Llama = field(default_factory=_model_loader)
    user_prompt_template: str = _default_user_prompt_template
    index: Index = field(default_factory=get_index)

    @property
    def system_prompt(self) -> str:
        return _default_system_prompt

    def query(self, question: str) -> str:
        """Generate an answer to the question, using the available knowledge.

        Using chat-completion of the instruct model to better control the output
        (via system prompt and EOT token)
        """
        return self._create_chat_completion(question)

    def _expand_user_prompt(self, question: str) -> str:
        """Infuse a question of the user with context"""
        context = "\n".join(self.index.query(question))
        prompt = self.user_prompt_template.format(context=context, question=question)
        logger.debug(f"built final prompt:\n{prompt}")
        return prompt

    def _create_completion(self, question: str) -> str:
        prompt = self._expand_user_prompt(question)
        # result = self.llm(prompt, max_tokens=config.model.max_completion_tokens)

        # return result["choices"][0]["text"]

    def _create_chat_completion(self, question: str) -> str:
        prompt = self._expand_user_prompt(question)
        max_tokens = config._select("model.max_completion_tokens", default=None)
        logger.debug(f"prompting model for chat completion for {max_tokens} tokens")
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