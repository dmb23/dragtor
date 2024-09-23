from dragtor.index import store


def test_store(empty_store: store.ChromaDBStore):
    assert "tests/assets" in empty_store._db_path
    assert empty_store.client.count_collections() == 1
    assert empty_store.collection.count() == 0
