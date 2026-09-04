#!/usr/bin/env python3
"""Verify that lossless representations induce identical reward evaluations."""
from strategic_agents.evaluation.regret import counterfactual_reward, price_grid
from strategic_agents.representations import compact, extract_semantics, redundant, verbose

canonical = {"current_round": 1, "last_prices": {"agent_0": 9.5},
    "last_profits": {"agent_0": 90.0}, "own_previous_prices": [9.5],
    "own_previous_profits": [90.0], "chosen_price": 9.5, "chosen_share": .2,
    "recent_history": [{"step": 0, "prices": {"agent_0": 9.5},
                        "rewards": {"agent_0": 90.0}, "done": False}]}
variants = {"compact": compact(canonical), "verbose": verbose(canonical),
            "redundant": redundant(canonical)}
reward_vectors = {}
for name, representation in variants.items():
    state = extract_semantics(representation, name)
    reward_vectors[name] = [counterfactual_reward(state["chosen_price"], state["chosen_share"], p)
                            for p in price_grid()]
assert reward_vectors["compact"] == reward_vectors["verbose"] == reward_vectors["redundant"]
print("PASS: 31/31 reward evaluations are identical across all lossless representations")
