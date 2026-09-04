"""Canonical semantic state behind all lossless Pricing serializations."""
from __future__ import annotations
import copy


def semantic_context(record: dict) -> dict:
    return copy.deepcopy(record)


def extract_semantics(context: dict, representation: str) -> dict:
    value = copy.deepcopy(context)
    if representation == "compact": return value
    if representation == "verbose":
        value["recent_history"] = [{"step": r["step"], "prices": {a: v["value"]["price"] for a, v in r["actions"].items()},
                                    "rewards": r["rewards"], "done": r["done"]} for r in value["recent_history"]]
        return value
    if representation == "redundant":
        duplicate = value.pop("redundant_equivalent_view")
        expected = {"current_market": {"round": value["current_round"], "last_prices": value["last_prices"],
                                       "last_profits": value["last_profits"]},
                    "trajectory_facts": value["recent_history"],
                    "focal_recent": {"prices": value["own_previous_prices"], "profits": value["own_previous_profits"]}}
        if duplicate != expected: raise ValueError("redundant view is not equivalent")
        return value
    raise ValueError("unknown representation")
