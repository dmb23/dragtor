![Dragtor Logo](assets/dragtor.png)

# Codename "dRAGtor"

This is a solution to get grounded LLM advice on climbing-related health and training issues.
All model interactions are meant to run locally, developed for an Mac M1.

> [!CAUTION]
> This is meant as a techncal project to work with different technologies, not to replace health professionals.
> Please do not rely on a machine with a lab coat and a stethoscope for your health questions!

## Getting Started
Again, mostly meant for experimentation. But allows to use via a CLI when installed via [Poetry](https://python-poetry.org/)

- have a python environment with Python >=3.11 and `poetry` for package management. A lock file is part of the repo.
- make sure `llama-cpp-python` runs with some sort of GPU / MPS support on your machine (or bring probably a lot of patience?)
- download a model and put it under /dragtor/models folder
    - I use [a quantized Llama3.1 8B Instruct](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf)
- to install locally, go to dragtor folder then run:
  - `python -m build`
  - `pip install -e [local_directory]` (e.g. /Users/[username]/dragtor/)
  - `python run_setup.py`
  To make sure all dependencies are fulfilled
- `poetry run dragtor` to start via the CLI

## Current Overview

I provide a small CLI to interact with the main functionality, most steps can be configured via `config/params.yml`

- Load data
    - load blog pages from [Hooper's Beta](www.hoopersbeta.com)
        - use Jina Reader API to easily parse the content
    - load audio data from [Nugget Climbing Podcast](https://thenuggetclimbing.com/episodes)
        - use whisper.cpp to transcribe, and 
        - pyannote.audio to diarize the conversation (differentiate the speakers). But this feature still not used 
- split data
- create vector embeddings
- store embeddings in a Vector Store
    - use [Chroma](https://www.trychroma.com/)
- use RAG to answer questions
    - use Llama3.1 8B as language model (in smaller quantizations)

## Open Issues

### small TODOs

- ✅ ~implement a proper prompt for Llama3.1~
- ✅ ~implement the correct distance function for retrieval~
- ✅ ~create good questions to be answered with the available information.~
    - build reference answers using the full text as context ( ~ 9750 tokens )

### Bugs

- ✅ ~ChromaDB creates duplicate entries when the same chunk is added multiple times~
- ✅ ~clean output of the LLM, especially hide the loading information (or place it in a debug message)~

### Managing context for Retrieval

The blogs from Hoopers beta are long-form on a single topic. Splitting those into small chunks makes it difficult to map those chunks to the original topic.

- ✅ Understand context length options for local LLMs to have a better feeling for my options
    - I can easily increase the context length, it does not seem to have any effect on load times or memory requirements (tested between 16 and 64000 tokens)
    - current model uses 40ms / 60ms for prompt eval / generation. I.e. 1s / 9 words prompt length; 1s / 5 words answer length.
- ✅ use cosine similarity instead of default L2 ☹️
- Look into alternative embeddings:
    -💡 Jina https://huggingface.co/jinaai/jina-embeddings-v2-base-en
    - Mixedbread https://huggingface.co/mixedbread-ai/deepset-mxbai-embed-de-large-v1
    - Stella https://huggingface.co/dunzhang/stella_en_400M_v5
- 💡 try other splitting strategies
    - try the usual RecursiveTextSplitter
- 💡 try to use an additional re-ranker
    - sentence transformers https://sbert.net/docs/cross_encoder/usage/usage.html
    - exists from Mixedbreak
- give access to a full blog?
- create LARGE chunks to answer questions.
    - check if using embeddings of large chunks works, or if embeddings should be of smaller sub-chunks.

### Agent Mode

It would be really powerful (and slow) to run in Agent mode with multiple steps.

- implement a simple two-step approach to understand structured output parsing
    - Model generates an answer in JSON -> This answer is used in a second step as input
- design a strategy to use multiple Model invokations to improve results
    - implement

#### experimental strategies
- generate embeddings from summaries of large parts of content?
    - embeddings for long-form text might be difficult to align with embeddings of questions
- JINA offers embeddings for up to 8K tokens - how well does this work?
- generate summaries of all entries (full blog posts), map those to the question, load the full article as context



## Possible Extensions

- Chat mode
    - use the chat endpoint
    - collect a history of message interactions
    - always feed a given number of recent messages into the request
        - e.g. "system, user, assistant, user" to be able to refer to the last 1 message

- Data loading
    - make it possible to extract all pages under `www.hoopersbeta.com/library/`
        - ... that have actual information
    - make it possible to load podcasts (e.g. The Nugget)
        - load the actual sound file, understand the APIs
        - use whisper.cpp to parse them to text
            - ideally use entity matching to separate into interviewer / interviewee
    - make it possible to use Youtube videos (e.g. Lattice, early Hoopers Beta)
        - download the sound only
        - use whisper.cpp to parse

lower priority / outdated ideas:

- Prompt mangement
    - dspy? 󰜴 apparently a framework to learn better prompts, requires metrics and test data
- Experiment management
    - MLFlow for tracking
        - store params, query, candidates, output
        - param to set if MLFlow should track or not
- ~Vector Embeddings~ 󰜴 at the moment I can see no advantage over Chroma, functionality is similar, speed / deployment is not an issue
    - switch to FAISS - what is the advantage over Chroma DB?
    - switch to DuckDB - there might still be issues with persistence?

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

### it is possible to generate images using FLUX.1 on a Laptop

yes, `dragtor.png` is generated on a Mac M1.


---

## Ideas:

- loading sentence_tranfsormers and querying an LLM model creates a segfault. Maybe downgrade torch?
    - super wierd: importing sentence_transformers in an imported sub-module creates a segfault. Importing it directly in llm.py does not.
    - it seems I can switch to transformers instead of sentence_transformers (thereby evading torch), and do not get the segfaults.

- use a model to split blogs into sessions
- use a model to extract pieces of information on relevant topics from a text
    - use a model to identify relevant topics for a given text
    - use a model to check if the model was making up pieces of information
