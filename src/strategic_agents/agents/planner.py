"""Semantic planner interface for hierarchical S2 policies."""
from typing import Callable


class SemanticPlanner:
    def __init__(self, plan: Callable[[dict], str]): self.plan = plan
    def intent(self, state_and_history: dict) -> str: return self.plan(state_and_history)
