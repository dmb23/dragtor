![Dragtor Logo](assets/dragtor.png)

# Codename "dRAGtor"

This is a solution to get grounded LLM advice on climbing-related health and training issues.
All model interactions are meant to run locally, developed for an Mac M1.

> [!CAUTION]
> This is meant as a techncal project to work with different technologies, not to replace health professionals.
> Please do not rely on a machine with a lab coat and a stethoscope for your health questions!

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

### small TODOs

- ✅ ~implement a proper prompt for Llama3.1~
- implement the correct distance function for retrieval
- create good questions to be answered with the available information.
    - build reference answers using the full text as context ( ~ 9750 tokens )

### Bugs

- ✅ ~ChromaDB creates duplicate entries when the same chunk is added multiple times~
- ✅ ~clean output of the LLM, especially hide the loading information (or place it in a debug message)~

### Managing context for Retrieval

The blogs from Hoopers beta are long-form on a single topic. Splitting those into small chunks makes it difficult to map those chunks to the original topic.

- ✅ Understand context length options for local LLMs to have a better feeling for my options
    - I can easily increase the context length, it does not seem to have any effect on load times or memory requirements (tested between 16 and 64000 tokens)
    - current model uses 40ms / 60ms for prompt eval / generation. I.e. 1s / 9 words prompt length; 1s / 5 words answer length.
- use cosine similarity instead of default L2 ☹️
- Look into alternative embeddings:
    - Jina https://huggingface.co/jinaai/jina-embeddings-v2-base-en
    - Mixedbread https://huggingface.co/mixedbread-ai/deepset-mxbai-embed-de-large-v1
    - Stella https://huggingface.co/dunzhang/stella_en_400M_v5
- try other splitting strategies
    - try the usual RecursiveTextSplitter
- try to use an additional re-ranker
    - sentence transformers https://sbert.net/docs/cross_encoder/usage/usage.html
    - exists from Mixedbreak

#### experimental strategies
- generate embeddings from summaries of large parts of content?
    - embeddings for long-form text might be difficult to align with embeddings of questions
- JINA offers embeddings for up to 8K tokens - how well does this work?
- generate summaries of all entries (full blog posts), map those to the question, load the full article as context



## Possible Extensions

- Data loading
    - make it possible to extract all pages under `www.hoopersbeta.com/library/`
- Vector Embeddings
    - switch to FAISS - what is the advantage over Chroma DB?
    - switch to DuckDB - there might still be issues with persistence?
- Prompt mangement
    - dspy?
- Experiment management
    - MLFlow for tracking
        - store params, query, candidates, output
        - param to set if MLFlow should track or not

## Learnings

### ~Change project structure to Kedro?~
tried that, it was very large overhead for little percieved advantage

- custom datasets for all data persistence
- overhead with Dataset -> catalog -> pipeline -> node -> actual functionality
- not made to be used with Poetry

### ✅ ~The LLM continues to generate after finishing the first answer~

- ✅ switch to the correct prompt template
    - using `generate_chat_response` works much better in
        - including a system prompt
        - finishing the answer at a desired location (when prescribed in system prompt)
