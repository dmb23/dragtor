from importlib.resources import as_file, files

from _pytest.logging import LogCaptureFixture
from dragtor import config
from dragtor.index import store
from loguru import logger
from omegaconf import OmegaConf
import pytest


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Send loguru logs to caplog"""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture(autouse=True, scope="session")
def mock_config():
    source = files("tests.assets").joinpath("test_params.yml")

    with as_file(source) as test_conf_path:
        test_config = OmegaConf.load(test_conf_path)

    config.conf.update(test_config)


@pytest.fixture
def example_text() -> str:
    return """
Intro

The A2 is one of the most important pulleys in our hands. It’s what allows us to hold onto those tiny edges without the other parts of our finger essentially just coming apart. But how? It’s just a piece of tissue holding another tissue in place. And why is tearing your A2 the most ubiquitous injury in climbing?

In this mega video/manual we’re going to go in-depth into the A2 pulley. We will talk about anatomy, the causes of a tear and how to test yourself for it, the things you can do to treat it as well as ways of preventing it from happening in the first place and in the future, and the thing everyone wants to know: how soon can you get back to climbing.

To fully understand this injury and how we can heal from it we need to get our bearings, anatomically speaking. So to start off we’re going to take a look at our fingers and how they actually work, which in turn will make the testing and treatments sections make a lot more sense.

Let’s dive in.

Anatomy

The flexor digitorum profundus (FDP) runs from the proximal ¾’s of the medial and anterior surfaces of the ulna and interosseous membrane to the base of distal phalanges of the 2nd through 5th digits.

When the FDP muscle contracts, the tendon retracts, which causes the finger to flex, just like if there was a string attached to the tip of the finger.

Now, what keeps the string, or tendon in this case, from bowstringing out away from the bones? That’s where your pulleys come into play. The A2 is one of five pulleys in each finger that holds the flexor tendon tight up against the bones. So when you flex your finger, the tendon slides back and forth under the pulley while the pulleys keep it in position.

So what happens when we actually hold something with our fingers, or in other words, apply force to them? I mean I don’t know about you guys but usually when I flex my fingers I’m not just doing it in the air for no reason.

Well the more load you place on your fingers, the more tensile force your flexor tendon has to withstand in order to keep your finger flexed and not allow it to open up.

Now remember that the pulleys are the ones keeping that flexor tendon tight up against the bones so it doesn’t spring out like a bowstring. So as you can imagine, if you continue to increase the force on the flexor tendon (in other words, put more weight on your fingers), you're going to be increasing the force on the pulleys.

Now you might be thinking, “Okay, so if you put too much force on your pulleys, you get injured. Simple!” And you’d be right, but there’s a bit more to it than that. So let’s move onto the next segment and I’ll explain, because what causes an A2 injury is actually really important for understanding how to avoid injuring it again.

Causes
"""


@pytest.fixture
def empty_store() -> store.ChromaDBStore:
    vstore = store.get_store()
    if vstore.client.count_collections() > 1:
        for collection in vstore.client.list_collections():
            if collection.name != vstore.collection.name:
                vstore.client.delete_collection(collection.name)
    if vstore.collection.count() > 0:
        vstore.collection.delete(vstore.collection.get()["ids"])

    return vstore


@pytest.fixture
def full_store(empty_store: store.ChromaDBStore, example_text: str) -> store.ChromaDBStore:
    empty_store.add_chunks(example_text.split("\n\n"))

    return empty_store
