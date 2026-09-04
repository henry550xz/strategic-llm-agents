#!/usr/bin/env python3
from strategic_agents.representations import compact, extract_semantics, redundant, verbose

state = {"current_round": 1, "last_prices": {"agent_0": 9.5}, "last_profits": {"agent_0": 20.0},
         "own_previous_prices": [9.5], "own_previous_profits": [20.0],
         "recent_history": [{"step": 0, "prices": {"agent_0": 9.5}, "rewards": {"agent_0": 20.0}, "done": False}]}
variants = {"compact": compact(state), "verbose": verbose(state), "redundant": redundant(state)}
assert all(extract_semantics(value, name) == state for name, value in variants.items())
print("PASS: all three serializations recover the identical canonical state")
