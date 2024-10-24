from dataclasses import dataclass, field
import hashlib


def ident(text: str) -> str:
    """Create a unique key for a text"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


@dataclass
class Messages:
    _parts: list = field(default_factory=list)

    def system(self, message: str):
        self._parts.append(("system", message))

    def user(self, message: str):
        self._parts.append(("user", message))

    def assistant(self, message: str):
        self._parts.append(("assistant", message))

    def format(self) -> list[dict]:
        messages = [{"role": role, "content": message} for role, message in self._parts]
        return messages
