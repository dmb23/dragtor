from dataclasses import dataclass, field

from llama_cpp import Llama

from dragtor.config import config
from dragtor.embed import ChromaDBIndex, Index


def _model_loader():
    return Llama(config.model.file_path, verbose=False, **config.model.kwargs)


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


"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ user_msg_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ model_answer_1 }}<|eot_id|>
"""
