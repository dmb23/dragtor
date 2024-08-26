from dragtor.llm import LlamaHandler, LocalDragtor


def test_llm_query():
    llm = LlamaHandler.from_config()
    with llm:
        res = llm.query_llm("What is the meaning of life?", n_predict=12)

    assert type(res) is str
    assert len(res) > 0


def test_gen_query():
    dr = LocalDragtor()
    res = dr.answer("What is the meaning of life?")

    assert type(res) is str
    assert len(res) > 0
