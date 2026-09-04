"""Interface for S1 direct language-model strategic policies."""
from typing import Callable


class DirectPolicy:
    def __init__(self, generate: Callable[[dict], dict]): self.generate = generate
    def act(self, state_and_history: dict) -> dict: return self.generate(state_and_history)
