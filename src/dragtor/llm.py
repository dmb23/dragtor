from dataclasses import dataclass, field

from llama_cpp import Llama

from dragtor.config import config
from dragtor.index.index import Index

# Llama prompting is included in the chat template when using Llama.cpp
_llama_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ user_prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

_default_system_prompt = "You are an assistant who provides advice on health care and training questions for climbers. Please answer questions in 3 paragraphs."

_default_user_prompt_template = """Please use the following pieces of context to answer the question.
context:
{context}

question:
{question}

answer:
"""


def _model_loader():
    return Llama(config.model.file_path, verbose=True, **config.model.kwargs)


@dataclass
class Generator:
    llm: Llama = field(default_factory=_model_loader)
    user_prompt_template: str = _default_user_prompt_template
    index: Index = field(default_factory=Index)

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
        context = self.index.query(question, 1)[0]
        return self.user_prompt_template.format(context=context, question=question)

    def _create_completion(self, question: str) -> str:
        prompt = self._expand_user_prompt(question)
        result = self.llm(prompt, max_tokens=config.model.max_tokens)

        return result["choices"][0]["text"]

    def _create_chat_completion(self, question: str) -> str:
        result = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": self._expand_user_prompt(question),
                },
            ]
        )

        return result["choices"][0]["message"]["content"]
