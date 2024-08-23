from dataclasses import dataclass
from pathlib import Path
import subprocess
from time import sleep

from llama_cpp import json
import requests


@dataclass
class LlamaHandler:
    modelpath: Path

    def _build_server_command(self) -> str:
        return f"llama-server -m {str(self.modelpath.resolve())} -ngl 15"

    def __enter__(self):
        self.p = subprocess.Popen(
            self._build_server_command(), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def __exit__(self, exc_type, exc_value, traceback):
        self.p.terminate()


modelpath = Path(
    "/Users/mischa/Projects/local/llama-cpp/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)
with LlamaHandler(modelpath) as m:
    sleep(1)
    do_continue = False
    try:
        res = requests.get(
            "http://127.0.0.1:8080/health", headers={"Content-Type": "application/json"}
        )
        print(res)
        do_continue = True
    except Exception as e:
        print(e)

    if do_continue:
        try:
            res = requests.post(
                "http://127.0.0.1:8080/completion",
                data=json.dumps(
                    {
                        "prompt": "what is the meaning of life?",
                        "n_predict": 32,
                    }
                ),
                headers={"Content-Type": "application/json"},
            )
            print(res)
            print(res.json())
        except Exception as e:
            print(e)
