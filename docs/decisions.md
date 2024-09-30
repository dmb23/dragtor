
- loading sentence_tranfsormers and querying an LLM model creates a segfault. Maybe downgrade torch?
    - super wierd: importing sentence_transformers in an imported sub-module creates a segfault. Importing it directly in llm.py does not.
    - it seems I can switch to transformers instead of sentence_transformers (thereby evading torch), and do not get the segfaults.

