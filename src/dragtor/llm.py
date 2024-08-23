from dataclasses import dataclass, field

from llama_cpp import Llama
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
class Generator:
    llm: Llama = field(default_factory=_model_loader)
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
        result = self.llm(prompt, max_tokens=config.model.max_completion_tokens)

        return result["choices"][0]["text"]

    def _create_chat_completion(self, question: str) -> str:
        prompt = self._expand_user_prompt(question)
        max_tokens = config._select("model.max_completion_tokens", default=None)
        logger.debug(f"prompting model for chat completion for {max_tokens} tokens")
        result = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            # max_tokens=max_tokens,
        )
        logger.debug("finished llm chat completion")

        return result["choices"][0]["message"]["content"]
