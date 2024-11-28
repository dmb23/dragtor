import marimo

__generated_with = "0.9.20"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    import numpy as np

    from dragtor.index.index import LateChunkingIndex
    from dragtor.data import get_all_loaders
    return LateChunkingIndex, get_all_loaders, mo, np


@app.cell
def __(LateChunkingIndex):
    lci = LateChunkingIndex()
    return (lci,)


@app.cell
def __(get_all_loaders):
    full_texts = sum([l.get_cache() for l in get_all_loaders()], start=[])
    text = full_texts[0]

    print(f"Number of characters: {len(text)}")
    return full_texts, text


@app.cell
def __(lci, np, text):
    chunks, chunk_annotations = lci.chunker.chunk_and_annotate(text)
    print(f"Chunk annotations span from {np.array(chunk_annotations).min()} to {np.array(chunk_annotations).max()}")
    return chunk_annotations, chunks


@app.cell
def __(chunk_annotations):
    chunk_annotations
    return


@app.cell
def __(lci, text):
    tokenized = lci.tokenizer(
        lci.embedding_model._task_instructions['retrieval.passage'] + text,
        return_tensors="pt",
        return_offsets_mapping=True
    )
    print(f"Number of tokens: {tokenized['input_ids'].shape[1]}")

    token_offsets = tokenized['offset_mapping'].squeeze().numpy() - 38
    print(f"tokens span characters from {token_offsets.min()} to {token_offsets.max()}")
    return token_offsets, tokenized


@app.cell
def __(token_offsets):
    token_offsets
    return


app._unparsable_cell(
    r"""
    # token_chunk_annotations = lci._map_chunks_to_tokens(chunk_annotations, token_offsets)

    ann = np.array(chunk_annotations)

    start_conditions = (token_offsets[:, 0][:, None] <= ann[:, 0]) & (token_offsets[:, 1][:, None] > ann[:, 0])
    start_tokens = np.argmax(start_conditions, axis=0)

    end_conditions = (token_offsets[:, 0][:, np.newaxis] < ann[:, 1]) & (jj
        token_offsets[:, 1][:, np.newaxis] >= ann[:, 1]
    )
    end_tokens = np.argmax(end_conditions, axis=0) + 1  # end_token is exclusive
    """,
    name="__"
)


@app.cell
def __(ann, np, start_tokens, token_offsets):
    print(start_tokens)

    print(np.argwhere(start_tokens==0).squeeze())

    chunk = ann[4]
    print(chunk)

    print(np.argwhere((token_offsets[:, 0] <= chunk[0]) & (token_offsets[:, 1] > chunk[1])))
    tstart = token_offsets[:, 0]
    tend = token_offsets[:, 1]
    return chunk, tend, tstart


@app.cell
def __(lci):
    prefix = lci.embedding_model._task_instructions['retrieval.passage']
    print(prefix)
    print(f"length of prefix: {len(prefix)}")

    prefix_tokens = lci.tokenizer(prefix, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    n_tokens = prefix_tokens['input_ids'].shape[1]
    print(f"n tokens of prefix: {n_tokens}")

    print(prefix_tokens['offset_mapping'])
    return n_tokens, prefix, prefix_tokens


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
