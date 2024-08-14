from dataclasses import dataclass, field

from llama_cpp import Llama

from dragtor.config import config
from dragtor.embed import ChromaDBIndex, Index


def _model_loader():
    return Llama(config.model.file_path, **config.model.kwargs)


@dataclass
class Generator:
    llm: Llama = field(default_factory=_model_loader)
    prompt_template: str = """Please use the following pieces of context to answer the question.
context:
{context}

question:
{question}

answer:
"""
    index: Index = field(default_factory=ChromaDBIndex)

    def query(self, question: str) -> str:
        context = self.index.query(question, 1)[0]
        prompt = self.prompt_template.format(question=question, context=context)
        result = self.llm(prompt, max_tokens=config.model.max_tokens)

        return result["choices"][0]["text"]
