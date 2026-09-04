#!/usr/bin/env python3
"""Train a small Kuhn CFR+ reference policy and enumerate decisions."""
from strategic_agents.envs.openspiel import cfr_plus_reference, enumerate_legal_decisions, load_game

try:
    game = load_game("kuhn_poker")
except ImportError as exc:
    raise SystemExit(str(exc))
policy, nash_conv = cfr_plus_reference(game, iterations=200)
decisions = enumerate_legal_decisions(game)
print({"game": "kuhn_poker", "cfr_plus_iterations": 200,
       "decision_nodes": len(decisions), "information_states": len({x["information_state"] for x in decisions}),
       "nash_conv": nash_conv, "all_legal": all(x["legal_actions"] for x in decisions)})
