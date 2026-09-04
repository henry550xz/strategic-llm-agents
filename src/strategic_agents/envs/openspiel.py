"""CPU-safe OpenSpiel utilities for imperfect-information games."""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache


def require_pyspiel():
    try:
        import pyspiel
    except ImportError as exc:
        raise ImportError("Install the optional dependency: pip install -e '.[openspiel]'") from exc
    return pyspiel


def load_game(name: str = "kuhn_poker"):
    return require_pyspiel().load_game(name)


def enumerate_legal_decisions(game, focal_player: int = 0) -> list[dict]:
    """Enumerate reachable focal-player decisions under full tree traversal."""
    rows = []
    def visit(state):
        if state.is_terminal(): return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes(): visit(state.child(action))
            return
        player = state.current_player()
        if player == focal_player:
            rows.append({"information_state": state.information_state_string(player),
                         "observation": state.observation_string(player),
                         "legal_actions": tuple(state.legal_actions())})
        for action in state.legal_actions(): visit(state.child(action))
    visit(game.new_initial_state())
    return rows


def cfr_plus_reference(game, iterations: int = 5000):
    """Train CFR+ and return its average policy and NashConv."""
    from open_spiel.python import policy
    from open_spiel.python.algorithms import cfr, exploitability
    solver = cfr.CFRPlusSolver(game)
    for _ in range(iterations): solver.evaluate_and_update_policy()
    average = solver.average_policy()
    return average, float(exploitability.nash_conv(game, average))


def exact_state_value(state, policy, focal_player: int = 0) -> float:
    """Exact continuation value under a reference policy."""
    @lru_cache(maxsize=None)
    def value(serialized: str):
        current = game.deserialize_state(serialized)
        if current.is_terminal(): return float(current.returns()[focal_player])
        if current.is_chance_node():
            return sum(p * value(current.child(a).serialize()) for a, p in current.chance_outcomes())
        probs = policy.action_probabilities(current, current.current_player())
        return sum(p * value(current.child(a).serialize()) for a, p in probs.items())
    game = state.get_game()
    return value(state.serialize())
