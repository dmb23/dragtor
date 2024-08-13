from dragtor import data, embed


def test_chroma_embedding():
    full_texts = data.JinaLoader().get_cache()

    chi = embed.ChromaDBIndex()
    chunks = chi.chunk_texts(full_texts)
    chi.embed_chunks(chunks)
