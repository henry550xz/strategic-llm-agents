import copy


def verbose(canonical: dict) -> dict:
    value = copy.deepcopy(canonical)
    value["recent_history"] = [{"step": r["step"], "actions": {a: {"action_type": "set_price", "agent_id": a,
                                "value": {"price": p}} for a, p in r["prices"].items()},
                                "rewards": r["rewards"], "done": r["done"]} for r in value["recent_history"]]
    return value
