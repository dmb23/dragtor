# Ideas and current Plan

## Current Plan

- ✅ figure out how to use [[#Model Checkpoints]] and decide how to continue for now:
    - ✅I can continue to use the `llama-server` command and manage slot state
    - ✅that means I can continue to use the easy interface via chat messages with better message ending
- ~figure out how conversation mode works with the `llama-cli` command~

- 💡 Allow for evaluation of RAG answers
    - ✅ faithfulness: 
        - extract propositions from an answer.
        - for each proposition:
            - check if proposition can be attributed to source context
    - 󰹏 AnswerCorrectness
        - extract propositions from an answer.
            - check how the proposition relates to a reference answer:
                - TP: proposition is present in reference
                - FP: proposition is not present in reference
                - FN: proposition from reference is not present in answer

- Finish Implementation
    - ✅ AnswerCorrectness
    - ✅ check: can I rewrite the `LlamaServerHandler` so that it detects if a server is up and wraps any call only if required? Maybe as a decorator, maybe in the relevant functions?
    - evaluate performance
        - 💡 make it end-to-end usable
        - create defined testcases
        - create evaluation summaries
        - revisit reference answers using the full text as context ( ~ 9750 tokens )
            - use Llama3.1-70B from Grok and tune the prompt for faithfulness?
    - create features
        - store features from script in VectorStore
    - late chunking

- improve chunking
    - late chunking in combination with semantic chunking sounds interesting
    - look into agentic chunking!

## Open Issues

### Model Checkpoints

It should be possible to save the model state to disk.

- [x] `llama-cli` allows to save state after an initial prompt prefix. This could store "system prompt", context, and possible few-shot prompts
    - I still do not understand how to properly delimit messages via `llama-cli`
- [x] `llama-server` allows to save state somehow
    - [x] ~check if that does what I expect it to~ That does what it is supposed to!
- 💡figure out how to decide on which context to load from the checkpoint
    - write a new cli command `preload` that
        - takes all cached documents from an index
        - creates state files for system prompt + basic context stuff + document
        - write summaries into the vector store
        - write summaries of all different topics into the vector store
    - add a method `chat_from_file` to a `LlamaHandler`
    - add a flag to `answer` that allows to either answer from RAG or from single file


### Load Podcast data

- [ ] create a new `DataLoader` to
    - [ ] load files from podcast feeds
    - [ ] parse them to text (e.g. whisper.cpp)
        - [ ] check how difficult it is to do "entity recognition" - Who is speaking?
    - [ ] store that information in plain text, ready for further processing


### small TODOs

- ✅ ~implement a proper prompt for Llama3.1~
- ✅ ~implement the correct distance function for retrieval~
- ✅ ~create good questions to be answered with the available information.~

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

## Ideas:

- use a model to split blogs into sessions
- use a model to extract pieces of information on relevant topics from a text
    - use a model to identify relevant topics for a given text
    - use a model to check if the model was making up pieces of information
