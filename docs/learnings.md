# Learnings

## ~Change project structure to Kedro?~
tried that, it was very large overhead for little percieved advantage

- custom datasets for all data persistence
- overhead with Dataset -> catalog -> pipeline -> node -> actual functionality
- not made to be used with Poetry

## ✅ ~The LLM continues to generate after finishing the first answer~

- ✅ switch to the correct prompt template
    - using `generate_chat_response` works much better in
        - including a system prompt
        - finishing the answer at a desired location (when prescribed in system prompt)

## it is possible to generate images using FLUX.1 on a Laptop

yes, `dragtor.png` is generated on a Mac M1.


