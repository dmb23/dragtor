![Dragtor Logo](assets/dragtor.png)

# Codename "dRAGtor"

This is a solution to get grounded LLM advice on climbing-related health and training issues.
All model interactions are meant to run locally, developed for an Mac M1.

> [!CAUTION]
> This is meant as a technical project to work with different technologies, not to replace health professionals.
> Please do not rely on a machine with a lab coat and a stethoscope for your health questions!

## Getting Started
Again, mostly meant for experimentation. But allows to use via a CLI when installed via [Poetry](https://python-poetry.org/)

- have a python environment with Pyton 3.11 and `uv` for package management. A lock file is part of the repo.
- make sure `llama-cpp` runs with some sort of GPU / MPS support on your machine (or bring probably a lot of patience?). This should be installed separately before, the different cli commands should be accessible in the Path
- download the models and put it under /dragtor/models folder
    - I use [a quantized Llama3.1 8B Instruct](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf)
    - For transcription, use [base EN model](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base.en.bin)
- `brew install ffmpeg` as it needed to convert audio format
  - [!!!] This is a global installation for Mac
- to run transcription feature, we need an executable file from [whisper.cpp](https://github.com/ggerganov/whisper.cpp):
  1. Clone the project into your local machine: `git clone https://github.com/ggerganov/whisper.cpp.git`
  2. Navigate into the directory: `cd whisper.cpp`
  3. Compile the project using the provided Makefile: `make base.en`
     - Transcription model can also be found under /whisper.cpp/models
     - You can also directly copy this model into /dragtor/models. Using Mac terminal: `cp models/ggml-base.en.bin /<your-path>/dragtor/models/`
  4. After a few minutes, find an executable file with name `main`
  5. Copy it to /dragtor folder, and rename it to `transcribe`
     - If you're using Mac terminal: `cp main /<your-path>/dragtor/transcribe`
- Add dragtor project path into your system path
  - For example, in ~/.zshrc (or another system file): `export PATH="$PATH:/<your-path>/dragtor`
- `uv run dragtor` to start via the CLI
- setup a secret credential to store secret keys: config/credentials.yml
```commandline
creds:
  jina: "jina_xxx"  # For JINA API
  hf: "hf_xxx"  # For huggingface secret key to use diarization model
```

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
