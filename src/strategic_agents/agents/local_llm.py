"""Thin S3 adapter; heavyweight Transformers loading is intentionally optional."""


class LocalLLMPolicy:
    def __init__(self, generator): self.generator = generator
    def act(self, state_and_history: dict) -> dict: return self.generator(state_and_history)
