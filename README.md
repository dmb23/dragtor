# Codename "dRAGtor"

I am building a solution to get grounded LLM advice on climbing-related health and training issues.

## Next steps:

- Data loading
    - write Python functionality to load information from Hoopers Beta
        - load data from Jina API
            - manage API key as secret?
- Vector Embeddings
    - start with FAISS in-memory DB
    - DuckDB + Files? Would be amazing!
- Easiest RAG:
    - get results
    - include in query
    - show result?

### Extensions
- Data loading
        - parse all pages, create possible configuration
        - configure which pages should be crawled
- Vector Embeddings
    - DuckDB
- Prompt mangement
    - dspy?
- Project structure - Kedro?
