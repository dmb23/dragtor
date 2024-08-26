from dragtor.llm import LlamaHandler


def test_llm_query():
    llm = LlamaHandler.from_config()
    with llm:
        res = llm.query("What is the meaning of life?", n_predict=12)

    assert type(res) is str
    assert len(res) > 0
