from collections import deque
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

from loguru import logger

from dragtor import config
from dragtor.index.index import Index, get_index
from dragtor.llm.llm import LlamaServerHandler


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
