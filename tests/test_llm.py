from dragtor.llm import LocalDragtor


def test_gen_query():
    dr = LocalDragtor()
    res = dr.answer("What is the meaning of life?")

    assert type(res) is str
    assert len(res) > 0


def test_chat():
    dr = LocalDragtor()
    res = dr.chat("What is the meaning of life?")

    assert type(res) is str
    assert len(res) > 0
