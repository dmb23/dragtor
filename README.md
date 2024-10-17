![Dragtor Logo](assets/dragtor.png)

# Codename "dRAGtor"

This is a solution to get grounded LLM advice on climbing-related health and training issues.
All model interactions are meant to run locally, developed for an Mac M1.

> [!CAUTION]
> This is meant as a techncal project to work with different technologies, not to replace health professionals.
> Please do not rely on a machine with a lab coat and a stethoscope for your health questions!

## Getting Started
Again, mostly meant for experimentation. But allows to use via a CLI when installed via [Poetry](https://python-poetry.org/)

- have a python environment with Pyton 3.11 and `uv` for package management. A lock file is part of the repo.
- make sure `llama-cpp` runs with some sort of GPU / MPS support on your machine (or bring probably a lot of patience?). This should be installed separately before, the different cli commands should be accessible in the Path
- download the models and put it under /dragtor/models folder
    - I use [a quantized Llama3.1 8B Instruct](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf)
    - For diarize (feature is not yet active), use [base EN model](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base.en.bin)
- `uv run dragtor` to start via the CLI
- install PyTorch, to use Mac-optimized packages:
    - `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/macos`
    - Or common packages: `pip install torch torchvision torchaudio`
  - `brew install ffmpeg` as it needed to convert audio format
    - [!!!] This is a global installation for Mac

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
