# Codename "dRAGtor"

`dRAGtor` is a personal project to get a better understanding of LLMs and RAG techniques.
I am building a solution to get grounded LLM advice on climbing-related health and training issues.
All model interactions are meant to run locally, developed for an Mac M1.

## Getting Started
... something something 

- `poetry`
- `llama-cpp-python`
- download a model
- `poetry run dragtor` 

## Current Overview

I provide a small CLI to interact with the main functionality, most steps can be configured via `config/params.yml`

- Load data
    - load blog pages from [Hooper's Beta](www.hoopersbeta.com)
        - use Jina Reader API to easily parse the content
- split data
- create vector embeddings
- use RAG to answer questions

## Open Issues

### Managing context for Retrieval

The blogs from Hoopers beta are long-form on a single topic. Splitting those into small chunks makes it difficult to map those chunks to the original topic.

- Understand context length options for local LLMs to have a better feeling for my options
- design better splitting strategy
    - larger chunks (but fewer)?
    - try to chunk by headings?
    - play around with using LLMs to generate summaries for querying?

### ChromaDB creates duplicate entries

... the heading says it all. It's a bug.

### The LLM continues to generate after finishing the first answer

Understand things like prompt template, stop tokens, repeat penalty, ...

## Possible Extensions

- Data loading
    - make it possible to extract all pages under `www.hoopersbeta.com/library/`
- Vector Embeddings
    - switch to FAISS - what is the advantage over Chroma DB?
    - switch to DuckDB - there might still be issues with persistence?
- Prompt mangement
    - dspy?

## Learnings

- ~Change project structure to Kedro?~
    - tried that, it was very large overhead for little percieved advantage
        - custom datasets for all data persistence
        - overhead with Dataset -> catalog -> pipeline -> node -> actual functionality
        - not made to be used with Poetry
